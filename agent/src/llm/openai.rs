use crate::llm;
use crate::{Error, Result};
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{
        ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessageArgs,
        ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
        ChatCompletionRequestToolMessage, ChatCompletionRequestToolMessageContent,
        ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
        ChatCompletionTool, ChatCompletionToolArgs, ChatCompletionToolType,
        CreateChatCompletionRequestArgs, FunctionCall, FunctionObjectArgs, Role, WebSearchOptions,
    },
};
use async_trait::async_trait;

pub struct OpenAI {
    model: String,
    client: Client<OpenAIConfig>,
}

impl OpenAI {
    pub fn new(model: String) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            model,
            client: Client::new(),
        })
    }
}

impl TryFrom<&llm::Message> for ChatCompletionRequestMessage {
    type Error = Error;

    fn try_from(msg: &llm::Message) -> Result<Self> {
        match msg {
            llm::Message::User(msg) => Ok(ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text(msg.clone()),
                    name: None,
                },
            )),
            llm::Message::System(msg) => Ok(ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessage {
                    content: ChatCompletionRequestSystemMessageContent::Text(msg.clone()),
                    name: None,
                },
            )),
            llm::Message::Tool { id, result, .. } => Ok(ChatCompletionRequestMessage::Tool(
                ChatCompletionRequestToolMessage {
                    content: ChatCompletionRequestToolMessageContent::Text(result.clone()),
                    tool_call_id: id.clone(),
                },
            )),
            llm::Message::Assistant(msg, tool_calls) => {
                Ok(ChatCompletionRequestMessage::Assistant(
                    ChatCompletionRequestAssistantMessageArgs::default()
                        .content(ChatCompletionRequestAssistantMessageContent::Text(
                            msg.clone(),
                        ))
                        .tool_calls(
                            tool_calls
                                .iter()
                                .map(|call| ChatCompletionMessageToolCall {
                                    id: call.id.clone(),
                                    r#type: ChatCompletionToolType::Function,
                                    function: FunctionCall {
                                        name: call.name.clone(),
                                        arguments: call.args.clone(),
                                    },
                                })
                                .collect::<Vec<_>>(),
                        )
                        .build()?,
                ))
            }
        }
    }
}

impl TryFrom<&llm::ToolDefinition> for ChatCompletionTool {
    type Error = Error;

    fn try_from(tool: &llm::ToolDefinition) -> Result<Self> {
        let res = ChatCompletionToolArgs::default()
            .function(
                FunctionObjectArgs::default()
                    .name(tool.name.clone())
                    .description(tool.desc.clone())
                    .parameters(tool.params.clone())
                    .build()?,
            )
            .build()?;

        Ok(res)
    }
}

#[async_trait]
impl llm::LLM for OpenAI {
    async fn completion<'a>(
        &self,
        request: llm::CompletionRequest<'a>,
    ) -> Result<llm::CompletionResponse> {
        let mut completion = CreateChatCompletionRequestArgs::default();
        completion
            .model(&self.model)
            .messages(
                request
                    .messages
                    .into_iter()
                    .map(ChatCompletionRequestMessage::try_from)
                    .collect::<Result<Vec<_>>>()?,
            )
            .tools(
                request
                    .tools
                    .into_iter()
                    .map(ChatCompletionTool::try_from)
                    .collect::<Result<Vec<_>>>()?,
            );

        if request.web_search_tool {
            completion.web_search_options(WebSearchOptions::default());
        }

        let completion = completion.build()?;

        let res = self.client.chat().create(completion).await?;

        if res.choices.is_empty() {
            return Err(Error::LLMResponseError("choices is empty".to_string()));
        }

        if res.choices[0].message.role != Role::User {
            return Err(Error::LLMResponseError(
                "expected role to be assistant".to_string(),
            ));
        }

        let content = res.choices[0]
            .message
            .content
            .as_ref()
            .ok_or(Error::LLMResponseError("content is empty".to_string()))?;

        let tool_calls = res.choices[0]
            .message
            .tool_calls
            .iter()
            .flat_map(|calls| {
                calls.iter().map(|call| llm::ToolCall {
                    id: call.id.clone(),
                    name: call.function.name.clone(),
                    args: call.function.arguments.clone(),
                })
            })
            .collect();

        Ok(llm::CompletionResponse {
            content: content.clone(),
            tool_calls,
        })
    }
}
