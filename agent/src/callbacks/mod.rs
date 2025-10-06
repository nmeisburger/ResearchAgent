use crate::Result;
use crate::llm::Message;
use crate::tools::SummarizeHistory;
use async_trait::async_trait;

#[async_trait]
pub trait Callback {
    async fn call(&mut self, messages: Vec<Message>) -> Result<Vec<Message>>;
}

#[async_trait]
impl Callback for SummarizeHistory {
    async fn call(&mut self, messages: Vec<Message>) -> Result<Vec<Message>> {
        self.summarize_history(messages).await
    }
}
