import pytest
from ai_project.conversation import ConversationContext


class TestConversationContext:
    """Test suite for conversation context manager"""
    
    @pytest.fixture
    def context(self):
        """Create context instance"""
        return ConversationContext(max_history=3)
    
    def test_initialization(self, context):
        """Test context initialization"""
        assert context.history == []
        assert context.max_history == 3
    
    def test_add_single_qa(self, context):
        """Test adding single Q&A pair"""
        context.add("What is Python?", "A programming language")
        
        assert len(context.history) == 1
        assert context.history[0]["question"] == "What is Python?"
        assert context.history[0]["answer"] == "A programming language"
    
    def test_add_multiple_qa(self, context):
        """Test adding multiple Q&A pairs"""
        context.add("Q1", "A1")
        context.add("Q2", "A2")
        context.add("Q3", "A3")
        
        assert len(context.history) == 3
    
    def test_max_history_limit(self, context):
        """Test that history respects max_history limit"""
        context.add("Q1", "A1")
        context.add("Q2", "A2")
        context.add("Q3", "A3")
        context.add("Q4", "A4")  # Should remove Q1
        
        assert len(context.history) == 3
        assert context.history[0]["question"] == "Q2"
        assert context.history[-1]["question"] == "Q4"
    
    def test_get_context_empty(self, context):
        """Test get_context with empty history"""
        result = context.get_context()
        assert result == ""
    
    def test_get_context_with_history(self, context):
        """Test get_context with history"""
        context.add("What is AI?", "Artificial Intelligence")
        context.add("What is ML?", "Machine Learning")
        
        result = context.get_context()
        
        assert "Previous conversation:" in result
        assert "Q: What is AI?" in result
        assert "A: Artificial Intelligence" in result
        assert "Q: What is ML?" in result
        assert "A: Machine Learning" in result
    
    def test_clear(self, context):
        """Test clearing history"""
        context.add("Q1", "A1")
        context.add("Q2", "A2")
        
        context.clear()
        
        assert context.history == []
        assert context.get_context() == ""
    
    def test_context_ordering(self, context):
        """Test that context maintains chronological order"""
        context.add("First", "1")
        context.add("Second", "2")
        context.add("Third", "3")
        
        result = context.get_context()
        
        # Check order in output
        first_pos = result.find("First")
        second_pos = result.find("Second")
        third_pos = result.find("Third")
        
        assert first_pos < second_pos < third_pos
    
    @pytest.mark.parametrize("max_history", [1, 5, 10])
    def test_different_max_history(self, max_history):
        """Test with different max_history values"""
        context = ConversationContext(max_history=max_history)
        
        # Add more than max_history
        for i in range(max_history + 5):
            context.add(f"Q{i}", f"A{i}")
        
        assert len(context.history) == max_history


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
