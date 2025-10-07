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
    messages: Vec<llm::Message>,
    tools: HashMap<String, Tool>,
    callbacks: Vec<Callback>,
    tool_defs: Vec<tools::ToolDefinition>,
    stop_condition: Box<dyn StopCondition + Send>,
    llm_websearch: bool,
}

impl Agent {
    async fn execute_tool_call(&mut self, tool_call: &tools::ToolCall) -> Result<()> {
        let tool = self
            .tools
            .get_mut(&tool_call.name)
            .ok_or(Error::ToolDoesNotExist(tool_call.name.clone()))?;

        let messages = std::mem::take(&mut self.messages);
        self.messages = tool.invoke(tool_call, messages).await?;

        Ok(())
    }

    pub async fn run(mut self) -> Result<Vec<Message>> {
        while !self.stop_condition.done(&self.messages) {
            let next = self
                .llm
                .completion(llm::CompletionRequest {
                    messages: &self.messages,
                    tools: &self.tool_defs,
                    web_search_tool: self.llm_websearch,
                })
                .await?;

            self.messages.push(llm::Message::Assistant(
                next.content,
                next.tool_calls.clone(),
            ));

            for tool_call in &next.tool_calls {
                self.execute_tool_call(tool_call).await?;
            }

            for callback in &mut self.callbacks {
                let messages = std::mem::take(&mut self.messages);
                self.messages = callback.call(messages).await?;
            }
        }

        Ok(self.messages)
    }
}

pub struct AgentBuilder {
    llm: Option<Arc<dyn llm::LLM + Send + Sync>>,
    system_prompt: Option<String>,
    user_prompt: Option<String>,
    tools: Vec<Tool>,
    callbacks: Vec<Callback>,
    stop_condition: Option<Box<dyn StopCondition + Send>>,
    llm_websearch: bool,
}

impl AgentBuilder {
    pub fn new() -> Self {
        Self {
            llm: None,
            system_prompt: None,
            user_prompt: None,
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

    pub fn system_prompt(mut self, prompt: String) -> Self {
        self.system_prompt = Some(prompt);
        self
    }

    pub fn user_prompt(mut self, prompt: String) -> Self {
        self.user_prompt = Some(prompt);
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

        let mut messages = Vec::new();
        if let Some(system) = self.system_prompt {
            messages.push(llm::Message::System(system));
        }
        if let Some(user) = self.user_prompt {
            messages.push(llm::Message::User(user));
        }

        if messages.is_empty() {
            return Err(Error::MissingArg(
                "system and/or user prompt is required for agent".to_string(),
            ));
        }

        Ok(Agent {
            llm: self
                .llm
                .ok_or(Error::MissingArg("llm is required for agent".to_string()))?,
            messages: messages,
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
