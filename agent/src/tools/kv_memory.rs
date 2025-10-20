use crate::Result;
use crate::llm::Message;
use crate::tools::{FunctionalTool, Tool, ToolCall, ToolDefinition};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct KVMemoryTool {
    memory: Arc<Mutex<HashMap<String, String>>>,
}

impl KVMemoryTool {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            memory: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    fn list_keys(&self) -> String {
        let mut s = "Keys in memory:\n".to_string();
        for key in self.memory.lock().unwrap().keys() {
            s.push_str(&format!("- {}\n", key));
        }
        s
    }

    fn get_key(&self, key: &str) -> String {
        self.memory
            .lock()
            .unwrap()
            .get(key)
            .map(|value| format!("value of key {}:\n{}", key, value))
            .unwrap_or(format!("key {} is not in memory", key))
    }

    fn set_key(&mut self, key: String, value: String) -> String {
        let msg = format!("key {} inserted into memory", key);
        self.memory.lock().unwrap().insert(key, value);
        msg
    }

    fn clone(&self) -> Self {
        Self {
            memory: self.memory.clone(),
        }
    }

    fn list_tool(&self) -> Box<MemoryListTool> {
        Box::new(MemoryListTool(self.clone()))
    }

    fn get_tool(&self) -> Box<MemoryGetTool> {
        Box::new(MemoryGetTool(self.clone()))
    }

    fn set_tool(&self) -> Box<MemorySetTool> {
        Box::new(MemorySetTool(self.clone()))
    }

    pub fn tools(&self) -> Result<Vec<Box<dyn Tool + Send>>> {
        Ok(vec![self.list_tool(), self.get_tool(), self.set_tool()])
    }
}

struct MemoryListTool(KVMemoryTool);

#[async_trait]
impl FunctionalTool for MemoryListTool {
    fn definition(&self) -> Result<super::ToolDefinition> {
        ToolDefinition::new::<()>(
            "memory_list_keys",
            "list the keys that are available in memory",
        )
    }

    async fn invoke_fn(&mut self, call: &ToolCall) -> Result<Message> {
        Ok(Message::Tool {
            id: call.id.clone(),
            name: "memory_list_keys".to_string(),
            result: self.0.list_keys(),
        })
    }
}

#[derive(Deserialize, JsonSchema)]
struct MemoryGetArgs {
    key: String,
}

struct MemoryGetTool(KVMemoryTool);

#[async_trait]
impl FunctionalTool for MemoryGetTool {
    fn definition(&self) -> Result<super::ToolDefinition> {
        ToolDefinition::new::<MemoryGetArgs>(
            "memory_get_key",
            "get the value associated with the given key in memory",
        )
    }

    async fn invoke_fn(&mut self, call: &ToolCall) -> Result<Message> {
        let args: MemoryGetArgs = call.args()?;
        Ok(Message::Tool {
            id: call.id.clone(),
            name: "memory_get_key".to_string(),
            result: self.0.get_key(&args.key),
        })
    }
}

#[derive(Deserialize, JsonSchema)]
struct MemorySetArgs {
    key: String,
    value: String,
}

struct MemorySetTool(KVMemoryTool);

#[async_trait]
impl FunctionalTool for MemorySetTool {
    fn definition(&self) -> Result<super::ToolDefinition> {
        ToolDefinition::new::<MemorySetArgs>(
            "memory_set_key",
            "set the value associated with the given key in memory",
        )
    }

    async fn invoke_fn(&mut self, call: &ToolCall) -> Result<Message> {
        let args: MemorySetArgs = call.args()?;
        Ok(Message::Tool {
            id: call.id.clone(),
            name: "memory_set_key".to_string(),
            result: self.0.set_key(args.key, args.value),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::KVMemoryTool;
    use crate::Result;
    use crate::llm::Message;
    use crate::tools::{FunctionalTool, ToolCall};

    async fn call_tool(tool: &mut dyn FunctionalTool, args: &str) -> Result<String> {
        match tool
            .invoke_fn(&ToolCall {
                id: String::new(),
                name: String::new(),
                args: args.to_string(),
            })
            .await?
        {
            Message::Tool { result, .. } => Ok(result),
            _ => panic!("not a tool message"),
        }
    }

    #[tokio::test]
    async fn test_kv_memory_tool() -> Result<()> {
        let kv_memory = KVMemoryTool::new();

        let mut list_tool = kv_memory.list_tool();
        let mut get_tool = kv_memory.get_tool();
        let mut set_tool = kv_memory.set_tool();

        assert_eq!(call_tool(&mut *list_tool, "").await?, "Keys in memory:\n");
        assert_eq!(
            call_tool(&mut *get_tool, "{\"key\":\"abc\"}").await?,
            "key abc is not in memory"
        );
        assert_eq!(
            call_tool(&mut *set_tool, "{\"key\":\"abc\",\"value\":\"123\"}").await?,
            "key abc inserted into memory"
        );

        assert_eq!(
            call_tool(&mut *list_tool, "").await?,
            "Keys in memory:\n- abc\n"
        );

        assert_eq!(
            call_tool(&mut *set_tool, "{\"key\":\"xyz\",\"value\":\"456\"}").await?,
            "key xyz inserted into memory"
        );

        assert_eq!(
            call_tool(&mut *get_tool, "{\"key\":\"abc\"}").await?,
            "value of key abc:\n123"
        );

        assert_eq!(
            call_tool(&mut *get_tool, "{\"key\":\"xyz\"}").await?,
            "value of key xyz:\n456"
        );

        assert_eq!(
            call_tool(&mut *set_tool, "{\"key\":\"abc\",\"value\":\"345\"}").await?,
            "key abc inserted into memory"
        );

        assert_eq!(
            call_tool(&mut *get_tool, "{\"key\":\"abc\"}").await?,
            "value of key abc:\n345"
        );

        Ok(())
    }
}
