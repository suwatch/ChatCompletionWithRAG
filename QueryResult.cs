namespace ChatCompletionWithRAG
{
    public class QuestionsResult
    {
        public string OriginalQuestion { get; set; }
        public string[] AlternativeQuestions { get; set; }
    }
}