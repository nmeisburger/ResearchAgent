use crate::Result;
use crate::llm::Message;
use async_trait::async_trait;
use schemars::{JsonSchema, schema_for};

mod kv_memory;
pub use kv_memory::KVMemoryTool;

mod summarize_history;
pub use summarize_history::SummarizeHistory;

pub struct ToolDefinition {
    pub name: String,
    pub desc: String,
    pub params: serde_json::Value,
}

impl ToolDefinition {
    pub fn new<P: JsonSchema>(name: &str, desc: &str) -> Result<Self> {
        let schema = schema_for!(P);
        let params = serde_json::to_value(&schema.schema)?;
        Ok(Self {
            name: name.to_string(),
            desc: desc.to_string(),
            params,
        })
    }
}

#[derive(Clone, std::hash::Hash)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: String,
}

impl ToolCall {
    pub fn args<O: for<'de> serde::Deserialize<'de>>(&self) -> Result<O> {
        let args = serde_json::from_str(&self.args)?;
        Ok(args)
    }
}

impl std::fmt::Display for ToolCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "- {} ({})\n\t- `{}\n`", self.name, self.id, self.args)
    }
}

#[async_trait]
pub trait Tool {
    fn definition(&self) -> Result<ToolDefinition>;

    async fn invoke(&mut self, args: &ToolCall, messages: Vec<Message>) -> Result<Vec<Message>>;
}

#[async_trait]
pub trait FunctionalTool {
    fn definition(&self) -> Result<ToolDefinition>;

    async fn invoke(&mut self, args: &ToolCall) -> Result<Message>;
}

#[async_trait]
impl<T> Tool for T
where
    T: FunctionalTool + Send + Sync,
{
    fn definition(&self) -> Result<ToolDefinition> {
        FunctionalTool::definition(self)
    }

    async fn invoke(
        &mut self,
        args: &ToolCall,
        mut messages: Vec<Message>,
    ) -> Result<Vec<Message>> {
        let result = FunctionalTool::invoke(self, args).await?;
        messages.push(result);
        Ok(messages)
    }
}
