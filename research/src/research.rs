use agent::llm;
use agent::llm::Message;
use agent::tools;
use agent::{Agent, AgentBuilder, StopCondition};
use agent::{Error, Result};
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
    pub fn new(llm: Arc<dyn llm::LLM + Send + Sync>) -> Result<Self> {
        let subagent_handles = Arc::new(Mutex::new(tokio::task::JoinSet::new()));

        Ok(Self {
            agent: AgentBuilder::new()
                .system_prompt(ORCHESTRATOR_PROMPT.to_string())
                .user_prompt("TODO".to_string())
                .llm(llm.clone())
                .llm_websearch()
                .tool(Box::new(CompleteTask))
                .tool(tools::SummarizeHistory::new(llm.clone(), 2))
                .tool(Box::new(StartSubAgent {
                    subagents: subagent_handles.clone(),
                    llm: llm.clone(),
                }))
                .tool(Box::new(WaitForSubAgent(subagent_handles)))
                .tools(tools::KVMemoryTool::new().tools()?)
                .callback(tools::SummarizeHistory::new(llm.clone(), 2))
                .stop_condition(Box::new(TaskCompleted))
                .build()?,
        })
    }

    pub async fn run(self) -> Result<Vec<Message>> {
        self.agent.run().await
    }
}

type SubAgentHandles = Arc<Mutex<tokio::task::JoinSet<Result<Vec<Message>>>>>;

struct StartSubAgent {
    subagents: SubAgentHandles,
    llm: Arc<dyn llm::LLM + Send + Sync>,
}

const ORCHESTRATOR_PROMPT: &str = include_str!("prompts/orchestrator.md");
const SUBAGENT_PROMPT: &str = include_str!("prompts/subagent.md");

#[async_trait]
impl tools::Tool for StartSubAgent {
    fn definition(&self) -> Result<tools::ToolDefinition> {
        tools::ToolDefinition::new::<String>(
            "start_subagent",
            "create a research sub-agent to investigate a specific research task",
        )
    }

    async fn invoke(
        &mut self,
        call: &tools::ToolCall,
        mut messages: Vec<Message>,
    ) -> Result<Vec<Message>> {
        let task: String = call.args()?;

        self.subagents.lock().await.spawn({
            let task = task.clone();
            let llm = self.llm.clone();
            async move {
                let agent = AgentBuilder::new()
                    .system_prompt(SUBAGENT_PROMPT.to_string())
                    .user_prompt(task.clone())
                    .llm(llm)
                    .llm_websearch()
                    .tool(Box::new(CompleteTask))
                    .tools(tools::KVMemoryTool::new().tools()?)
                    .stop_condition(Box::new(TaskCompleted))
                    .build()?;

                agent.run().await
            }
        });

        messages.push(Message::Tool {
            id: call.id.clone(),
            name: "start_subagent".to_string(),
            result: format!("Research sub-agent started for task: {}", task),
        });
        Ok(messages)
    }
}

struct WaitForSubAgent(SubAgentHandles);

#[async_trait]
impl tools::FunctionalTool for WaitForSubAgent {
    fn definition(&self) -> Result<tools::ToolDefinition> {
        tools::ToolDefinition::new::<()>(
            "wait_for_subagent",
            "wait for a research sub-agent to completes its task and obtain the result",
        )
    }

    async fn invoke(&mut self, call: &tools::ToolCall) -> Result<Message> {
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
            "finish your task and return the result to the lead researcher",
        )
    }

    async fn invoke(&mut self, call: &tools::ToolCall) -> Result<Message> {
        let result: String = call.args()?;

        Ok(Message::Tool {
            id: call.id.clone(),
            name: "complete_task".to_string(),
            result: result,
        })
    }
}
