mod research;
use agent::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let llm = agent::llm::OpenAI::new("gpt-4o".to_string());

    let orchestrator = research::Orchestrator::new(
        llm,
        "TODO Task".to_string(),
        &std::path::Path::new("todo log dir"),
    )?;

    orchestrator.run().await?;

    Ok(())
}
