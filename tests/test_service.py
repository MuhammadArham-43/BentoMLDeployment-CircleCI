from service import LLMService, CompletionResponse

def test_generate():
    service = LLMService()
    result = service.generate("Say hello")
    assert isinstance(result, CompletionResponse)
    assert len(result.completion) > 0
    assert "hello" in result.completion.lower()