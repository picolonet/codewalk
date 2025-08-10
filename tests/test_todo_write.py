import unittest
import sys
import os

# Add the cw directory to the path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cw'))

from cw_task import TodoItem, todo_write, get_todo_list_status


class TestTodoWrite(unittest.TestCase):
    """Unit tests for the TodoWrite functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sample_todos = [
            {
                "id": "1",
                "content": "Explore project structure and entry points",
                "status": "pending",
                "priority": "high"
            },
            {
                "id": "2", 
                "content": "Identify main application files and routing",
                "status": "pending",
                "priority": "high"
            },
            {
                "id": "3",
                "content": "Trace middleware and dependency injection flow",
                "status": "in_progress",
                "priority": "medium"
            },
            {
                "id": "4",
                "content": "Document the complete request handling flow",
                "status": "completed",
                "priority": "medium"
            }
        ]
    
    def test_todo_item_creation(self):
        """Test TodoItem class creation and validation."""
        todo = TodoItem(
            id="test_1",
            content="Test todo item",
            status="pending",
            priority="high"
        )
        
        self.assertEqual(todo.id, "test_1")
        self.assertEqual(todo.content, "Test todo item")
        self.assertEqual(todo.status, "pending")
        self.assertEqual(todo.priority, "high")
    
    def test_todo_item_to_dict(self):
        """Test TodoItem to_dict method."""
        todo = TodoItem(
            id="test_1",
            content="Test todo item",
            status="pending", 
            priority="high"
        )
        
        expected_dict = {
            "id": "test_1",
            "content": "Test todo item",
            "status": "pending",
            "priority": "high"
        }
        
        self.assertEqual(todo.to_dict(), expected_dict)
    
    def test_todo_write_success(self):
        """Test successful todo_write with valid input."""
        result = todo_write(self.sample_todos)
        
        self.assertIn("Successfully processed 4 todos", result)
        self.assertIn("2 pending", result)
        self.assertIn("1 in progress", result) 
        self.assertIn("1 completed", result)
        self.assertIn("2 high", result)
        self.assertIn("2 medium", result)
        self.assertIn("0 low", result)
    
    def test_todo_write_missing_field(self):
        """Test todo_write with missing required field."""
        invalid_todos = [
            {
                "id": "1",
                "content": "Missing status field",
                "priority": "high"
                # Missing "status" field
            }
        ]
        
        result = todo_write(invalid_todos)
        self.assertIn("Error: Todo item 1 missing required field 'status'", result)
    
    def test_todo_write_invalid_status(self):
        """Test todo_write with invalid status value."""
        invalid_todos = [
            {
                "id": "1",
                "content": "Test todo",
                "status": "invalid_status",
                "priority": "high"
            }
        ]
        
        result = todo_write(invalid_todos)
        self.assertIn("Error: Todo item 1 has invalid status 'invalid_status'", result)
        self.assertIn("Valid statuses: pending, in_progress, completed", result)
    
    def test_todo_write_invalid_priority(self):
        """Test todo_write with invalid priority value."""
        invalid_todos = [
            {
                "id": "1",
                "content": "Test todo",
                "status": "pending",
                "priority": "invalid_priority"
            }
        ]
        
        result = todo_write(invalid_todos)
        self.assertIn("Error: Todo item 1 has invalid priority 'invalid_priority'", result)
        self.assertIn("Valid priorities: high, medium, low", result)
    
    def test_todo_write_non_dict_item(self):
        """Test todo_write with non-dictionary item."""
        invalid_todos = [
            "This is not a dictionary"
        ]
        
        result = todo_write(invalid_todos)
        self.assertIn("Error: Todo item 1 is not a dictionary", result)
    
    def test_todo_write_empty_list(self):
        """Test todo_write with empty todo list."""
        result = todo_write([])
        
        self.assertIn("Successfully processed 0 todos", result)
    
    def test_get_todo_list_status(self):
        """Test get_todo_list_status function."""
        status = get_todo_list_status(self.sample_todos)
        
        expected_status = {
            "total_todos": 4,
            "status_counts": {
                "pending": 2,
                "in_progress": 1,
                "completed": 1
            },
            "priority_counts": {
                "high": 2,
                "medium": 2,
                "low": 0
            },
            "completion_rate": 25.0  # 1 out of 4 completed
        }
        
        self.assertEqual(status, expected_status)
    
    def test_get_todo_list_status_empty(self):
        """Test get_todo_list_status with empty list."""
        status = get_todo_list_status([])
        
        expected_status = {
            "total_todos": 0,
            "status_counts": {
                "pending": 0,
                "in_progress": 0,
                "completed": 0
            },
            "priority_counts": {
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "completion_rate": 0.0
        }
        
        self.assertEqual(status, expected_status)
    
    def test_get_todo_list_status_all_completed(self):
        """Test get_todo_list_status with all todos completed."""
        completed_todos = [
            {
                "id": "1",
                "content": "Completed task 1",
                "status": "completed",
                "priority": "high"
            },
            {
                "id": "2", 
                "content": "Completed task 2",
                "status": "completed",
                "priority": "low"
            }
        ]
        
        status = get_todo_list_status(completed_todos)
        self.assertEqual(status["completion_rate"], 100.0)
        self.assertEqual(status["status_counts"]["completed"], 2)
        self.assertEqual(status["status_counts"]["pending"], 0)
        self.assertEqual(status["status_counts"]["in_progress"], 0)
    
    def test_todo_write_with_example_input(self):
        """Test todo_write with the exact example input from the prompt."""
        example_input = [
            {
                "id": "1",
                "content": "Explore project structure and entry points",
                "status": "pending",
                "priority": "high"
            },
            {
                "id": "2",
                "content": "Identify main application files and routing",
                "status": "pending", 
                "priority": "high"
            },
            {
                "id": "3",
                "content": "Trace middleware and dependency injection flow",
                "status": "pending",
                "priority": "medium"
            },
            {
                "id": "4",
                "content": "Document the complete request handling flow",
                "status": "pending",
                "priority": "medium"
            }
        ]
        
        result = todo_write(example_input)
        
        # Should process successfully
        self.assertIn("Successfully processed 4 todos", result)
        self.assertIn("4 pending", result)
        self.assertIn("0 in progress", result)
        self.assertIn("0 completed", result)
        self.assertIn("2 high", result)
        self.assertIn("2 medium", result)
        self.assertIn("0 low", result)


if __name__ == '__main__':
    unittest.main()