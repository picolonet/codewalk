import unittest
import sys
import os
from datetime import datetime

# Add the cw directory to the path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cw'))

from cw_task import TaskPlan, TaskStatus, update_plan


class TestTaskPlan(unittest.TestCase):
    """Unit tests for TaskPlan and related functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.plan_steps = [
            "Analyze requirements",
            "Design architecture", 
            "Implement features",
            "Write tests",
            "Deploy to production"
        ]
        self.task_plan = TaskPlan(
            plan_name="Website Development",
            plan_description="Build a modern web application",
            plan_steps=self.plan_steps
        )
    
    def test_task_plan_initialization(self):
        """Test TaskPlan initialization."""
        self.assertEqual(self.task_plan.plan_name, "Website Development")
        self.assertEqual(self.task_plan.plan_description, "Build a modern web application")
        self.assertEqual(self.task_plan.plan_steps, self.plan_steps)
        
        # All steps should start as PENDING
        for step in self.plan_steps:
            self.assertEqual(self.task_plan.plan_status[step], TaskStatus.PENDING)
        
        # Timestamps should be set
        self.assertIsInstance(self.task_plan.created_at, datetime)
        self.assertIsInstance(self.task_plan.updated_at, datetime)
    
    def test_update_step_status_success(self):
        """Test successful step status update."""
        step = "Analyze requirements"
        result = self.task_plan.update_step_status(step, TaskStatus.IN_PROGRESS)
        
        self.assertTrue(result)
        self.assertEqual(self.task_plan.get_step_status(step), TaskStatus.IN_PROGRESS)
        
        # Updated timestamp should be changed
        self.assertGreaterEqual(self.task_plan.updated_at, self.task_plan.created_at)
    
    def test_update_step_status_failure(self):
        """Test step status update with non-existent step."""
        result = self.task_plan.update_step_status("Non-existent step", TaskStatus.COMPLETED)
        
        self.assertFalse(result)
    
    def test_get_step_status(self):
        """Test getting step status."""
        step = "Design architecture"
        
        # Initially pending
        status = self.task_plan.get_step_status(step)
        self.assertEqual(status, TaskStatus.PENDING)
        
        # After update
        self.task_plan.update_step_status(step, TaskStatus.COMPLETED)
        status = self.task_plan.get_step_status(step)
        self.assertEqual(status, TaskStatus.COMPLETED)
        
        # Non-existent step
        status = self.task_plan.get_step_status("Non-existent")
        self.assertIsNone(status)
    
    def test_get_all_steps_with_status(self):
        """Test getting all steps with their status."""
        # Update some steps
        self.task_plan.update_step_status("Analyze requirements", TaskStatus.COMPLETED)
        self.task_plan.update_step_status("Design architecture", TaskStatus.IN_PROGRESS)
        
        steps_with_status = self.task_plan.get_all_steps_with_status()
        
        # Should return all steps in order
        self.assertEqual(len(steps_with_status), len(self.plan_steps))
        
        # Check specific statuses
        step_dict = dict(steps_with_status)
        self.assertEqual(step_dict["Analyze requirements"], TaskStatus.COMPLETED)
        self.assertEqual(step_dict["Design architecture"], TaskStatus.IN_PROGRESS)
        self.assertEqual(step_dict["Implement features"], TaskStatus.PENDING)
    
    def test_to_xml_basic(self):
        """Test XML serialization with basic data."""
        xml_output = self.task_plan.to_xml()
        
        # Should be valid XML string
        self.assertIsInstance(xml_output, str)
        self.assertIn('<?xml version=', xml_output)
        self.assertIn('<TaskPlan>', xml_output)
        self.assertIn('</TaskPlan>', xml_output)
        
        # Should contain plan data
        self.assertIn('Website Development', xml_output)
        self.assertIn('Build a modern web application', xml_output)
        
        # Should contain all steps
        for step in self.plan_steps:
            self.assertIn(step, xml_output)
        
        # Should contain status attributes
        self.assertIn('status="pending"', xml_output)
    
    def test_to_xml_with_status_updates(self):
        """Test XML serialization after status updates."""
        # Update some statuses
        self.task_plan.update_step_status("Analyze requirements", TaskStatus.COMPLETED)
        self.task_plan.update_step_status("Design architecture", TaskStatus.IN_PROGRESS)
        self.task_plan.update_step_status("Write tests", TaskStatus.FAILED)
        
        xml_output = self.task_plan.to_xml()
        
        # Should contain different status values
        self.assertIn('status="completed"', xml_output)
        self.assertIn('status="in_progress"', xml_output)
        self.assertIn('status="failed"', xml_output)
        self.assertIn('status="pending"', xml_output)  # Remaining steps
    
    def test_xml_structure(self):
        """Test XML structure is well-formed."""
        import xml.etree.ElementTree as ET
        
        xml_output = self.task_plan.to_xml()
        
        # Should parse without errors
        root = ET.fromstring(xml_output)
        
        # Check structure
        self.assertEqual(root.tag, 'TaskPlan')
        
        # Check required elements exist
        name_elem = root.find('name')
        self.assertIsNotNone(name_elem)
        self.assertEqual(name_elem.text, 'Website Development')
        
        description_elem = root.find('description')
        self.assertIsNotNone(description_elem)
        
        steps_elem = root.find('steps')
        self.assertIsNotNone(steps_elem)
        
        # Check step elements
        step_elements = steps_elem.findall('step')
        self.assertEqual(len(step_elements), len(self.plan_steps))


class TestUpdatePlanFunction(unittest.TestCase):
    """Unit tests for the update_plan tool function."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.task_plan = TaskPlan(
            plan_name="Test Plan",
            plan_description="Test Description",
            plan_steps=["Step 1", "Step 2", "Step 3"]
        )
    
    def test_update_plan_success(self):
        """Test successful plan update."""
        result = update_plan(self.task_plan, "Step 1", "in_progress")
        
        self.assertIn("Successfully updated", result)
        self.assertEqual(self.task_plan.get_step_status("Step 1"), TaskStatus.IN_PROGRESS)
    
    def test_update_plan_invalid_step(self):
        """Test plan update with invalid step."""
        result = update_plan(self.task_plan, "Non-existent Step", "completed")
        
        self.assertIn("Error: Step 'Non-existent Step' not found", result)
    
    def test_update_plan_invalid_status(self):
        """Test plan update with invalid status."""
        result = update_plan(self.task_plan, "Step 1", "invalid_status")
        
        self.assertIn("Error: Invalid status", result)
        self.assertIn("Valid statuses are:", result)
    
    def test_update_plan_all_valid_statuses(self):
        """Test updating with all valid status values."""
        valid_statuses = ["pending", "in_progress", "completed", "failed"]
        
        for i, status in enumerate(valid_statuses):
            step = f"Step {i+1}"
            if i < len(self.task_plan.plan_steps):  # Only test existing steps
                result = update_plan(self.task_plan, step, status)
                self.assertIn("Successfully updated", result)
    
    def test_update_plan_case_insensitive(self):
        """Test that status update is case insensitive."""
        result = update_plan(self.task_plan, "Step 1", "COMPLETED")
        
        self.assertIn("Successfully updated", result)
        self.assertEqual(self.task_plan.get_step_status("Step 1"), TaskStatus.COMPLETED)


if __name__ == '__main__':
    unittest.main()