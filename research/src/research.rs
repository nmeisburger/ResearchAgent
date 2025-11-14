use agent::llm::Message;
use agent::tools;
use agent::{Agent, AgentBuilder, StopCondition};
use agent::{Error, Result};
use agent::{callbacks, llm};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex;

struct TaskCompleted;

impl StopCondition for TaskCompleted {
    fn done(&self, history: &[llm::Message]) -> bool {
        if let Some(llm::Message::Tool { name, .. }) = history.last() {
            return name == "complete_task";
        }
        false
    }
}

pub struct Orchestrator {
    agent: Agent,
}

impl Orchestrator {
    pub fn new(llm: Arc<dyn llm::LLM + Send + Sync>, log_dir: &std::path::Path) -> Result<Self> {
        let subagent_handles = Arc::new(Mutex::new(tokio::task::JoinSet::new()));

        let file = std::fs::File::create(log_dir.join("orchestrator.md"))?;

        Ok(Self {
            agent: AgentBuilder::new()
                // .system_prompt(ORCHESTRATOR_PROMPT.to_string())
                // .user_prompt(task_desc)
                .llm(llm.clone())
                .llm_websearch()
                .tool(Box::new(CompleteTask))
                .tool(tools::SummarizeHistory::new(llm.clone(), 2))
                .tool(Box::new(StartSubAgent {
                    subagents: subagent_handles.clone(),
                    llm: llm.clone(),
                    subagent_id: std::sync::atomic::AtomicU32::new(0),
                    log_dir: log_dir.to_path_buf(),
                }))
                .tool(Box::new(WaitForSubAgent(subagent_handles)))
                .tools(tools::KVMemoryTool::new().tools()?)
                .callback(tools::SummarizeHistory::new(llm.clone(), 2))
                .callback(callbacks::MessageLogger::new("orchestrator", file)?)
                .stop_condition(Box::new(TaskCompleted))
                .build()?,
        })
    }

    pub async fn run(mut self, task_desc: String) -> Result<String> {
        let mut history = self
            .agent
            .run(vec![
                Message::System(ORCHESTRATOR_PROMPT.to_string()),
                Message::User(task_desc),
            ])
            .await?;

        match history.pop() {
            Some(Message::Tool { name, result, .. }) if name == "complete_task" => Ok(result),
            Some(Message::Tool { name, .. }) => Err(Error::AgentWorkflowError(format!(
                "expected final message to be complete_task tool call not {} tool call",
                name
            ))),
            Some(_) => Err(Error::AgentWorkflowError(
                "expected final message to be complete_task tool call".to_string(),
            )),
            None => Err(Error::AgentWorkflowError(format!("message history empty"))),
        }
    }
}

type SubAgentHandles = Arc<Mutex<tokio::task::JoinSet<Result<Vec<Message>>>>>;

struct StartSubAgent {
    subagent_id: std::sync::atomic::AtomicU32,
    subagents: SubAgentHandles,
    llm: Arc<dyn llm::LLM + Send + Sync>,
    log_dir: std::path::PathBuf,
}

const ORCHESTRATOR_PROMPT: &str = include_str!("prompts/orchestrator.md");
const SUBAGENT_PROMPT: &str = include_str!("prompts/subagent.md");

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct StartSubAgentArgs {
    /// this is the description of the task that the sub-agent should complete
    task_desc: String,
}

#[async_trait]
impl tools::Tool for StartSubAgent {
    fn definition(&self) -> Result<tools::ToolDefinition> {
        tools::ToolDefinition::new::<StartSubAgentArgs>(
            "start_subagent",
            "This tool allows you to create a research sub-agent to investigate a specific research task. You must use this tool to delegate parts of your research task to sub-agents. Make sure to provide clear instructions to the sub-agent as to what it should research.",
        )
    }

    async fn invoke(
        &mut self,
        call: &tools::ToolCall,
        mut messages: Vec<Message>,
    ) -> Result<Vec<Message>> {
        let args: StartSubAgentArgs = call.args()?;

        self.subagents.lock().await.spawn({
            let name = format!(
                "subagent_{}",
                self.subagent_id
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            );
            let task_prompt = args.task_desc.clone();
            let llm = self.llm.clone();

            let file = std::fs::File::create(self.log_dir.join(format!("{}.md", name)))?;
            async move {
                let mut agent = AgentBuilder::new()
                    .llm(llm.clone())
                    .llm_websearch()
                    .tool(Box::new(CompleteTask))
                    .tool(tools::SummarizeHistory::new(llm.clone(), 2))
                    .tools(tools::KVMemoryTool::new().tools()?)
                    .callback(tools::SummarizeHistory::new(llm.clone(), 2))
                    .callback(callbacks::MessageLogger::new(&name, file)?)
                    .stop_condition(Box::new(TaskCompleted))
                    .build()?;

                agent
                    .run(vec![
                        Message::System(SUBAGENT_PROMPT.to_string()),
                        Message::User(task_prompt.clone()),
                    ])
                    .await
            }
        });

        messages.push(Message::Tool {
            id: call.id.clone(),
            name: "start_subagent".to_string(),
            result: format!("Research sub-agent started for task: {}", args.task_desc),
        });
        Ok(messages)
    }

    async fn on_agent_start(&mut self) -> Result<()> {
        self.subagents.lock().await.shutdown().await;
        Ok(())
    }
}

struct WaitForSubAgent(SubAgentHandles);

#[async_trait]
impl tools::FunctionalTool for WaitForSubAgent {
    fn definition(&self) -> Result<tools::ToolDefinition> {
        tools::ToolDefinition::new::<()>(
            "wait_for_subagent",
            "This tool will wait for any of the active sub-agents to complete, and return the result they provide for their completed task.",
        )
    }

    async fn invoke_fn(&mut self, call: &tools::ToolCall) -> Result<Message> {
        let mut result = match self.0.lock().await.join_next().await {
            Some(messages) => messages??,
            None => return Ok(Message::Tool {
                id: call.id.clone(),
                name: "wait_for_subagent".to_string(),
                result:
                    "no sub-agents are currently active, create a new sub-agent to wait for a task"
                        .to_string(),
            }),
        };

        if let Some(Message::Tool { name, result, .. }) = result.pop() {
            if name == "return_task_result" {
                return Ok(Message::Tool {
                    id: call.id.clone(),
                    name: "wait_for_subagent".to_string(),
                    result: result,
                });
            }
        }

        Err(Error::AgentWorkflowError(
            "sub agent terminated without correct tool call".to_string(),
        ))
    }
}

struct CompleteTask;

#[async_trait]
impl tools::FunctionalTool for CompleteTask {
    fn definition(&self) -> Result<tools::ToolDefinition> {
        tools::ToolDefinition::new::<String>(
            "complete_task",
            "This tool will mark your task as complete and return the result. You must use this tool when you have completed your task.",
        )
    }

    async fn invoke_fn(&mut self, call: &tools::ToolCall) -> Result<Message> {
        let result: String = call.args()?;

        Ok(Message::Tool {
            id: call.id.clone(),
            name: "complete_task".to_string(),
            result: result,
        })
    }
}
