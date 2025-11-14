use crate::agent::llm::Message;
use crate::callbacks;
use crate::llm;
use crate::tools;
use crate::{Error, Result};
use std::collections::HashMap;
use std::sync::Arc;

pub trait StopCondition {
    fn done(&self, history: &[llm::Message]) -> bool;
}

type Tool = Box<dyn tools::Tool + Send>;
type Callback = Box<dyn callbacks::Callback + Send>;

pub struct Agent {
    llm: Arc<dyn llm::LLM + Send + Sync>,
    tools: HashMap<String, Tool>,
    callbacks: Vec<Callback>,
    tool_defs: Vec<tools::ToolDefinition>,
    stop_condition: Box<dyn StopCondition + Send>,
    llm_websearch: bool,
}

impl Agent {
    async fn execute_tool_call(
        &mut self,
        tool_call: &tools::ToolCall,
        messages: Vec<llm::Message>,
    ) -> Result<Vec<llm::Message>> {
        let tool = self
            .tools
            .get_mut(&tool_call.name)
            .ok_or(Error::ToolDoesNotExist(tool_call.name.clone()))?;

        let messages = tool.invoke(tool_call, messages).await?;

        Ok(messages)
    }

    pub async fn run(&mut self, mut messages: Vec<llm::Message>) -> Result<Vec<Message>> {
        while !self.stop_condition.done(&messages) {
            let next = self
                .llm
                .completion(llm::CompletionRequest {
                    messages: &messages,
                    tools: &self.tool_defs,
                    web_search_tool: self.llm_websearch,
                })
                .await?;

            messages.push(llm::Message::Assistant(
                next.content,
                next.tool_calls.clone(),
            ));

            for tool_call in &next.tool_calls {
                messages = self.execute_tool_call(tool_call, messages).await?;
            }

            for callback in &mut self.callbacks {
                messages = callback.call(messages).await?;
            }
        }

        Ok(messages)
    }
}

pub struct AgentBuilder {
    llm: Option<Arc<dyn llm::LLM + Send + Sync>>,
    tools: Vec<Tool>,
    callbacks: Vec<Callback>,
    stop_condition: Option<Box<dyn StopCondition + Send>>,
    llm_websearch: bool,
}

impl AgentBuilder {
    pub fn new() -> Self {
        Self {
            llm: None,
            tools: Vec::new(),
            callbacks: Vec::new(),
            stop_condition: None,
            llm_websearch: false,
        }
    }

    pub fn llm(mut self, llm: Arc<dyn llm::LLM + Send + Sync>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools.extend(tools);
        self
    }

    pub fn callback(mut self, callback: Callback) -> Self {
        self.callbacks.push(callback);
        self
    }

    pub fn stop_condition(mut self, cond: Box<dyn StopCondition + Send>) -> Self {
        self.stop_condition = Some(cond);
        self
    }

    pub fn llm_websearch(mut self) -> Self {
        self.llm_websearch = true;
        self
    }

    pub fn build(self) -> Result<Agent> {
        let mut tool_defs = Vec::new();
        let mut tools = HashMap::new();

        for tool in self.tools {
            let def = tool.definition()?;
            tools.insert(def.name.clone(), tool);
            tool_defs.push(def);
        }

        Ok(Agent {
            llm: self
                .llm
                .ok_or(Error::MissingArg("llm is required for agent".to_string()))?,
            tools: tools,
            tool_defs: tool_defs,
            callbacks: self.callbacks,
            stop_condition: self.stop_condition.ok_or(Error::MissingArg(
                "stop_condition is required for agent".to_string(),
            ))?,
            llm_websearch: self.llm_websearch,
        })
    }
}

#[cfg(test)]
mod tests {
    use core::panic;

    use crate::llm::{CompletionRequest, CompletionResponse, LLM, Message};
    use crate::tools::{FunctionalTool, ToolCall, ToolDefinition};
    use crate::{AgentBuilder, Result, StopCondition};
    use async_trait::async_trait;
    use std::sync::Arc;

    struct MockLLM;

    #[async_trait]
    impl LLM for MockLLM {
        async fn completion<'a>(
            &self,
            request: CompletionRequest<'a>,
        ) -> Result<CompletionResponse> {
            match request.messages.last() {
                Some(Message::User(_)) => Ok(CompletionResponse {
                    content: "tool call".to_string(),
                    tool_calls: vec![ToolCall {
                        id: "call1".to_string(),
                        name: "double".to_string(),
                        args: "{\"arg\":123}".to_string(),
                    }],
                }),
                Some(Message::Tool { .. }) => Ok(CompletionResponse {
                    content: "tool call recieved".to_string(),
                    tool_calls: vec![],
                }),
                Some(Message::Assistant(_, _)) => Ok(CompletionResponse {
                    content: "completed".to_string(),
                    tool_calls: vec![],
                }),
                _ => panic!("unexpected message sequence"),
            }
        }
    }

    struct DoubleTool;

    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct DoubleArgs {
        arg: i32,
    }

    #[async_trait]
    impl FunctionalTool for DoubleTool {
        fn definition(&self) -> Result<ToolDefinition> {
            ToolDefinition::new::<DoubleArgs>("double", "double")
        }

        async fn invoke_fn(&mut self, tool_call: &ToolCall) -> Result<Message> {
            let args: DoubleArgs = tool_call.args()?;
            Ok(Message::Tool {
                id: tool_call.id.clone(),
                name: "double".to_string(),
                result: format!("2 * {} = {}", args.arg, 2 * args.arg),
            })
        }
    }

    struct SimpleStop;

    impl StopCondition for SimpleStop {
        fn done(&self, history: &[Message]) -> bool {
            if let Some(Message::Assistant(content, _)) = history.last() {
                content == "completed"
            } else {
                false
            }
        }
    }

    #[tokio::test]
    async fn test_agent() -> Result<()> {
        let mut agent = AgentBuilder::new()
            .llm(Arc::new(MockLLM))
            .tool(Box::new(DoubleTool))
            .stop_condition(Box::new(SimpleStop))
            .build()?;

        let history = agent
            .run(vec![Message::User("do stuff".to_string())])
            .await?;

        assert_eq!(history.len(), 5);

        assert!(matches!(&history[0], Message::User (content) if content == "do stuff"));
        assert!(matches!(&history[1], Message::Assistant (_, tool_calls) if tool_calls.len() == 1));
        assert!(matches!(&history[2], Message::Tool {  result,.. } if result == "2 * 123 = 246"));
        assert!(
            matches!(&history[3], Message::Assistant (content, _) if content== "tool call recieved")
        );
        assert!(matches!(&history[4], Message::Assistant (content, _) if content== "completed"));

        Ok(())
    }
}
