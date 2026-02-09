class ConversationContext:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history
    
    def add(self, question, answer):
        """Add Q&A pair to history"""
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_context(self):
        """Get formatted conversation history"""
        if not self.history:
            return ""
        
        context = "\n\nPrevious conversation:\n"
        for qa in self.history:
            context += f"Q: {qa['question']}\nA: {qa['answer']}\n"
        return context
    
    def clear(self):
        """Clear conversation history"""
        self.history = []
