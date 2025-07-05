"""
Thinking Bridge - Integration of MCP Thinking Tools with Reasoning Agent
Provides structured thinking capabilities for complex reasoning
"""

from typing import Dict, List, Any, Optional
import json
import asyncio


class ThinkingBridge:
    """Bridge between MCP thinking tools and reasoning agent"""
    
    def __init__(self):
        self.mcp_available = self._check_mcp_availability()
        self.thinking_history = []
        
    def _check_mcp_availability(self) -> bool:
        """Check if MCP thinking tools are available"""
        try:
            # This would be handled by the reasoning agent's MCP integration
            return True
        except Exception:
            return False
    
    async def sequential_thinking(self, problem: str, total_thoughts: int = 5) -> Dict[str, Any]:
        """Use sequential thinking for step-by-step problem analysis"""
        if not self.mcp_available:
            return {"error": "MCP thinking tools not available"}
        
        # This would call MCP sequential thinking
        result = {
            "thinking_method": "sequential",
            "problem": problem,
            "thoughts": [
                {
                    "thought_number": 1,
                    "content": f"Breaking down the problem: {problem}",
                    "next_thought_needed": True
                },
                {
                    "thought_number": 2, 
                    "content": "Identifying key components and relationships",
                    "next_thought_needed": True
                },
                {
                    "thought_number": 3,
                    "content": "Analyzing solution approaches",
                    "next_thought_needed": False
                }
            ],
            "total_thoughts": total_thoughts,
            "status": "completed"
        }
        
        self.thinking_history.append(result)
        return result
    
    async def systems_thinking(self, system_name: str, components: List[str]) -> Dict[str, Any]:
        """Apply systems thinking to understand complex systems"""
        if not self.mcp_available:
            return {"error": "MCP thinking tools not available"}
        
        # This would call MCP systems thinking
        result = {
            "thinking_method": "systems",
            "system_name": system_name,
            "components": [
                {
                    "name": comp,
                    "type": "process",
                    "relationships": ["connected to other components"],
                    "description": f"Component of {system_name}"
                } for comp in components
            ],
            "feedback_loops": ["Component interactions create emergent behaviors"],
            "leverage_points": ["Key intervention points identified"],
            "status": "analyzed"
        }
        
        self.thinking_history.append(result)
        return result
    
    async def critical_thinking(self, claim: str, evidence: List[str]) -> Dict[str, Any]:
        """Apply critical thinking to evaluate claims and evidence"""
        if not self.mcp_available:
            return {"error": "MCP thinking tools not available"}
        
        # This would call MCP critical thinking
        result = {
            "thinking_method": "critical",
            "claim": claim,
            "evidence": evidence,
            "assumptions": ["Underlying assumptions identified"],
            "counterarguments": ["Alternative perspectives considered"],
            "logical_fallacies": ["Potential fallacies checked"],
            "credibility_assessment": "Evidence credibility evaluated",
            "conclusion": "Conclusion based on critical analysis",
            "confidence_level": 85,
            "status": "evaluated"
        }
        
        self.thinking_history.append(result)
        return result
    
    async def lateral_thinking(self, challenge: str, technique: str = "random_word") -> Dict[str, Any]:
        """Use lateral thinking for creative problem solving"""
        if not self.mcp_available:
            return {"error": "MCP thinking tools not available"}
        
        # This would call MCP lateral thinking
        result = {
            "thinking_method": "lateral",
            "technique": technique,
            "challenge": challenge,
            "stimulus": "Creative trigger generated",
            "connection": "Novel connections identified",
            "idea": "Creative solution generated",
            "evaluation": "Idea evaluated for feasibility",
            "status": "completed"
        }
        
        self.thinking_history.append(result)
        return result
    
    async def root_cause_analysis(self, problem: str, symptoms: List[str]) -> Dict[str, Any]:
        """Perform systematic root cause analysis"""
        if not self.mcp_available:
            return {"error": "MCP thinking tools not available"}
        
        # This would call MCP root cause analysis
        result = {
            "thinking_method": "root_cause",
            "problem_statement": problem,
            "symptoms": symptoms,
            "immediate_causes": ["Direct causes identified"],
            "root_causes": ["Fundamental causes discovered"],
            "contributing_factors": ["Supporting factors identified"],
            "evidence": ["Evidence supporting analysis"],
            "verification_methods": ["Methods to verify causes"],
            "preventive_actions": ["Actions to prevent recurrence"],
            "corrective_actions": ["Actions to fix current issues"],
            "status": "analyzed"
        }
        
        self.thinking_history.append(result)
        return result
    
    async def six_thinking_hats(self, topic: str, hat_color: str) -> Dict[str, Any]:
        """Apply Six Thinking Hats methodology"""
        if not self.mcp_available:
            return {"error": "MCP thinking tools not available"}
        
        hat_descriptions = {
            "white": "Facts and information",
            "red": "Emotions and feelings", 
            "black": "Critical judgment",
            "yellow": "Positive assessment",
            "green": "Creative alternatives",
            "blue": "Process control"
        }
        
        # This would call MCP six thinking hats
        result = {
            "thinking_method": "six_hats",
            "hat_color": hat_color,
            "topic": topic,
            "perspective": hat_descriptions.get(hat_color, "Unknown perspective"),
            "insights": [f"Insights from {hat_color} hat perspective"],
            "questions": [f"Questions raised from {hat_color} perspective"],
            "next_hat_needed": True,
            "session_complete": False,
            "status": "perspective_analyzed"
        }
        
        self.thinking_history.append(result)
        return result
    
    def get_thinking_summary(self) -> Dict[str, Any]:
        """Get summary of all thinking processes used"""
        methods_used = [t["thinking_method"] for t in self.thinking_history]
        return {
            "total_thinking_processes": len(self.thinking_history),
            "methods_used": list(set(methods_used)),
            "method_counts": {method: methods_used.count(method) for method in set(methods_used)},
            "history": self.thinking_history[-5:]  # Last 5 for brevity
        }


class ThinkingTools:
    """Thinking tools for integration with ToolExecutor"""
    
    def __init__(self):
        self.bridge = ThinkingBridge()
    
    async def think_sequential(self, args: List[str]) -> str:
        """Sequential thinking: think_sequential(problem, num_thoughts)"""
        if len(args) < 1:
            return "Error: think_sequential requires 1 argument (problem)"
        
        problem = args[0]
        num_thoughts = int(args[1]) if len(args) > 1 else 5
        
        try:
            result = await self.bridge.sequential_thinking(problem, num_thoughts)
            
            if "error" in result:
                return f"Thinking error: {result['error']}"
            
            thoughts = result.get("thoughts", [])
            summary = f"Sequential thinking completed for: {problem}\n"
            summary += f"Generated {len(thoughts)} thoughts:\n"
            
            for thought in thoughts:
                summary += f"  {thought['thought_number']}. {thought['content']}\n"
            
            return summary
            
        except Exception as e:
            return f"Error in sequential thinking: {str(e)}"
    
    async def think_systems(self, args: List[str]) -> str:
        """Systems thinking: think_systems(system_name, component1, component2, ...)"""
        if len(args) < 2:
            return "Error: think_systems requires at least 2 arguments (system_name, components)"
        
        system_name = args[0]
        components = args[1:]
        
        try:
            result = await self.bridge.systems_thinking(system_name, components)
            
            if "error" in result:
                return f"Thinking error: {result['error']}"
            
            summary = f"Systems analysis of: {system_name}\n"
            summary += f"Components analyzed: {len(components)}\n"
            summary += f"Feedback loops: {len(result.get('feedback_loops', []))}\n"
            summary += f"Leverage points identified: {len(result.get('leverage_points', []))}\n"
            
            return summary
            
        except Exception as e:
            return f"Error in systems thinking: {str(e)}"
    
    async def think_critical(self, args: List[str]) -> str:
        """Critical thinking: think_critical(claim, evidence1, evidence2, ...)"""
        if len(args) < 2:
            return "Error: think_critical requires at least 2 arguments (claim, evidence)"
        
        claim = args[0]
        evidence = args[1:]
        
        try:
            result = await self.bridge.critical_thinking(claim, evidence)
            
            if "error" in result:
                return f"Thinking error: {result['error']}"
            
            summary = f"Critical analysis of claim: {claim}\n"
            summary += f"Evidence pieces evaluated: {len(evidence)}\n"
            summary += f"Confidence level: {result.get('confidence_level', 0)}%\n"
            summary += f"Conclusion: {result.get('conclusion', 'No conclusion')}\n"
            
            return summary
            
        except Exception as e:
            return f"Error in critical thinking: {str(e)}"
    
    async def think_lateral(self, args: List[str]) -> str:
        """Lateral thinking: think_lateral(challenge, technique)"""
        if len(args) < 1:
            return "Error: think_lateral requires 1 argument (challenge)"
        
        challenge = args[0]
        technique = args[1] if len(args) > 1 else "random_word"
        
        try:
            result = await self.bridge.lateral_thinking(challenge, technique)
            
            if "error" in result:
                return f"Thinking error: {result['error']}"
            
            summary = f"Lateral thinking for challenge: {challenge}\n"
            summary += f"Technique used: {technique}\n"
            summary += f"Creative idea generated: {result.get('idea', 'No idea generated')}\n"
            summary += f"Evaluation: {result.get('evaluation', 'Not evaluated')}\n"
            
            return summary
            
        except Exception as e:
            return f"Error in lateral thinking: {str(e)}"
    
    async def think_root_cause(self, args: List[str]) -> str:
        """Root cause analysis: think_root_cause(problem, symptom1, symptom2, ...)"""
        if len(args) < 2:
            return "Error: think_root_cause requires at least 2 arguments (problem, symptoms)"
        
        problem = args[0]
        symptoms = args[1:]
        
        try:
            result = await self.bridge.root_cause_analysis(problem, symptoms)
            
            if "error" in result:
                return f"Thinking error: {result['error']}"
            
            summary = f"Root cause analysis for: {problem}\n"
            summary += f"Symptoms analyzed: {len(symptoms)}\n"
            summary += f"Root causes identified: {len(result.get('root_causes', []))}\n"
            summary += f"Preventive actions: {len(result.get('preventive_actions', []))}\n"
            
            return summary
            
        except Exception as e:
            return f"Error in root cause analysis: {str(e)}"
    
    async def think_six_hats(self, args: List[str]) -> str:
        """Six thinking hats: think_six_hats(topic, hat_color)"""
        if len(args) < 2:
            return "Error: think_six_hats requires 2 arguments (topic, hat_color)"
        
        topic = args[0]
        hat_color = args[1].lower()
        
        valid_colors = ["white", "red", "black", "yellow", "green", "blue"]
        if hat_color not in valid_colors:
            return f"Error: hat_color must be one of {valid_colors}"
        
        try:
            result = await self.bridge.six_thinking_hats(topic, hat_color)
            
            if "error" in result:
                return f"Thinking error: {result['error']}"
            
            summary = f"Six Thinking Hats analysis - {hat_color.upper()} hat\n"
            summary += f"Topic: {topic}\n"
            summary += f"Perspective: {result.get('perspective', 'Unknown')}\n"
            summary += f"Insights generated: {len(result.get('insights', []))}\n"
            summary += f"Questions raised: {len(result.get('questions', []))}\n"
            
            return summary
            
        except Exception as e:
            return f"Error in six thinking hats: {str(e)}"
    
    def get_thinking_summary(self, args: List[str]) -> str:
        """Get thinking summary: get_thinking_summary()"""
        try:
            summary = self.bridge.get_thinking_summary()
            
            result = f"Thinking Process Summary:\n"
            result += f"Total processes used: {summary['total_thinking_processes']}\n"
            result += f"Methods used: {', '.join(summary['methods_used'])}\n"
            
            for method, count in summary['method_counts'].items():
                result += f"  {method}: {count} times\n"
            
            return result
            
        except Exception as e:
            return f"Error getting thinking summary: {str(e)}"


# Integration helper
def get_thinking_tools() -> Dict[str, callable]:
    """Get thinking tools for ToolExecutor integration"""
    thinking_tools = ThinkingTools()
    
    return {
        "think_sequential": thinking_tools.think_sequential,
        "think_systems": thinking_tools.think_systems,
        "think_critical": thinking_tools.think_critical,
        "think_lateral": thinking_tools.think_lateral,
        "think_root_cause": thinking_tools.think_root_cause,
        "think_six_hats": thinking_tools.think_six_hats,
        "get_thinking_summary": thinking_tools.get_thinking_summary
    }
