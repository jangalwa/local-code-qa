import pytest
from pathlib import Path


def load_skill(skill_path):
    """Load skill from SKILL.md file"""
    try:
        with open(skill_path, 'r') as f:
            return f.read()
    except:
        return ""


class TestSkillLoading:
    """Test skill loading functionality"""
    
    def test_load_existing_skill(self):
        """Test loading the Python expert skill"""
        skill_path = 'skills/python-expert/SKILL.md'
        skill = load_skill(skill_path)
        
        assert skill != ""
        assert "Python" in skill
        assert "expert" in skill.lower()
    
    def test_load_nonexistent_skill(self):
        """Test loading a non-existent skill returns empty string"""
        skill = load_skill('skills/nonexistent/SKILL.md')
        assert skill == ""
    
    def test_skill_contains_key_sections(self):
        """Test that skill has expected content"""
        skill_path = 'skills/python-expert/SKILL.md'
        skill = load_skill(skill_path)
        
        # Check for key sections
        assert "Type Hints" in skill or "type hint" in skill.lower()
        assert "Error Handling" in skill or "error" in skill.lower()
        assert "Performance" in skill or "performance" in skill.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
