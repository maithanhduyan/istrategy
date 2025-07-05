"""
Inference Engine for Reasoning Agent
Provides logical reasoning, deduction, and pattern recognition capabilities
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import re
import json
import itertools
from datetime import datetime
import asyncio


class LogicalInferenceEngine:
    """Engine for logical reasoning and inference"""
    
    def __init__(self):
        self.rules = []
        self.facts = []
        self.inference_history = []
    
    def add_rule(self, premise: str, conclusion: str, confidence: float = 1.0):
        """Add a logical rule: if premise then conclusion"""
        rule = {
            "id": len(self.rules),
            "premise": premise,
            "conclusion": conclusion,
            "confidence": confidence,
            "created_at": datetime.now().isoformat()
        }
        self.rules.append(rule)
        return rule["id"]
    
    def add_fact(self, fact: str, confidence: float = 1.0):
        """Add a known fact"""
        fact_obj = {
            "id": len(self.facts),
            "statement": fact,
            "confidence": confidence,
            "created_at": datetime.now().isoformat()
        }
        self.facts.append(fact_obj)
        return fact_obj["id"]
    
    def forward_chaining(self, max_iterations: int = 10) -> List[Dict]:
        """Apply forward chaining to derive new facts"""
        new_facts = []
        iteration = 0
        
        while iteration < max_iterations:
            derived_something = False
            
            for rule in self.rules:
                premise = rule["premise"]
                conclusion = rule["conclusion"]
                
                # Check if premise matches any fact
                for fact in self.facts:
                    if self._matches_pattern(fact["statement"], premise):
                        # Check if conclusion already exists
                        conclusion_exists = any(
                            self._matches_pattern(f["statement"], conclusion) 
                            for f in self.facts + new_facts
                        )
                        
                        if not conclusion_exists:
                            new_fact = {
                                "statement": conclusion,
                                "confidence": min(fact["confidence"], rule["confidence"]),
                                "derived_from": {
                                    "rule_id": rule["id"],
                                    "fact_id": fact["id"],
                                    "iteration": iteration
                                }
                            }
                            new_facts.append(new_fact)
                            derived_something = True
            
            if not derived_something:
                break
            
            iteration += 1
        
        # Add new facts to knowledge base
        for fact in new_facts:
            self.add_fact(fact["statement"], fact["confidence"])
        
        return new_facts
    
    def backward_chaining(self, goal: str) -> Dict[str, Any]:
        """Use backward chaining to prove a goal"""
        proof_steps = []
        
        def prove_goal(target_goal: str, depth: int = 0) -> bool:
            if depth > 10:  # Prevent infinite recursion
                return False
            
            # Check if goal is already a known fact
            for fact in self.facts:
                if self._matches_pattern(fact["statement"], target_goal):
                    proof_steps.append({
                        "step": len(proof_steps),
                        "type": "fact",
                        "statement": fact["statement"],
                        "depth": depth
                    })
                    return True
            
            # Check if goal can be derived from rules
            for rule in self.rules:
                if self._matches_pattern(rule["conclusion"], target_goal):
                    if prove_goal(rule["premise"], depth + 1):
                        proof_steps.append({
                            "step": len(proof_steps),
                            "type": "rule_application",
                            "rule": rule,
                            "depth": depth
                        })
                        return True
            
            return False
        
        proved = prove_goal(goal)
        
        return {
            "goal": goal,
            "proved": proved,
            "proof_steps": proof_steps,
            "steps_count": len(proof_steps)
        }
    
    def _matches_pattern(self, statement: str, pattern: str) -> bool:
        """Simple pattern matching for logical statements"""
        # For now, use simple string matching
        # In a more sophisticated system, this would handle variables and unification
        return statement.lower().strip() == pattern.lower().strip()
    
    def analyze_consistency(self) -> Dict[str, Any]:
        """Analyze knowledge base for consistency"""
        contradictions = []
        
        # Look for direct contradictions
        for i, fact1 in enumerate(self.facts):
            for j, fact2 in enumerate(self.facts[i+1:], i+1):
                if self._are_contradictory(fact1["statement"], fact2["statement"]):
                    contradictions.append({
                        "fact1": fact1,
                        "fact2": fact2,
                        "type": "direct_contradiction"
                    })
        
        return {
            "consistent": len(contradictions) == 0,
            "contradictions": contradictions,
            "total_facts": len(self.facts),
            "total_rules": len(self.rules)
        }
    
    def _are_contradictory(self, statement1: str, statement2: str) -> bool:
        """Check if two statements are contradictory"""
        # Simple contradiction detection
        # Look for "not X" vs "X" patterns
        if "not " in statement1.lower() and statement1.lower().replace("not ", "") == statement2.lower():
            return True
        if "not " in statement2.lower() and statement2.lower().replace("not ", "") == statement1.lower():
            return True
        return False


class PatternRecognitionEngine:
    """Engine for pattern recognition and analysis"""
    
    def __init__(self):
        self.patterns = []
        self.data_sequences = []
    
    def add_sequence(self, sequence: List[Any], label: str = "") -> int:
        """Add a data sequence for pattern analysis"""
        seq_obj = {
            "id": len(self.data_sequences),
            "sequence": sequence,
            "label": label,
            "length": len(sequence),
            "created_at": datetime.now().isoformat()
        }
        self.data_sequences.append(seq_obj)
        return seq_obj["id"]
    
    def find_numeric_patterns(self, sequence: List[Union[int, float]]) -> Dict[str, Any]:
        """Find patterns in numeric sequences"""
        if len(sequence) < 3:
            return {"error": "Sequence too short for pattern analysis"}
        
        patterns_found = []
        
        # Check for arithmetic progression
        differences = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
        if len(set(differences)) == 1:
            patterns_found.append({
                "type": "arithmetic_progression",
                "common_difference": differences[0],
                "confidence": 1.0
            })
        
        # Check for geometric progression
        if all(x != 0 for x in sequence[:-1]):
            ratios = [sequence[i+1] / sequence[i] for i in range(len(sequence)-1)]
            if len(set(ratios)) == 1:
                patterns_found.append({
                    "type": "geometric_progression", 
                    "common_ratio": ratios[0],
                    "confidence": 1.0
                })
        
        # Check for fibonacci-like pattern
        if len(sequence) >= 3:
            fibonacci_matches = 0
            for i in range(2, len(sequence)):
                if abs(sequence[i] - (sequence[i-1] + sequence[i-2])) < 0.001:
                    fibonacci_matches += 1
            
            if fibonacci_matches == len(sequence) - 2:
                patterns_found.append({
                    "type": "fibonacci_sequence",
                    "confidence": 1.0
                })
        
        # Check for polynomial patterns (simple quadratic)
        if len(sequence) >= 4:
            second_differences = []
            for i in range(len(differences)-1):
                second_differences.append(differences[i+1] - differences[i])
            
            if len(set(second_differences)) == 1:
                patterns_found.append({
                    "type": "quadratic_sequence",
                    "second_difference": second_differences[0],
                    "confidence": 0.9
                })
        
        return {
            "sequence": sequence,
            "patterns_found": patterns_found,
            "pattern_count": len(patterns_found)
        }
    
    def find_text_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """Find patterns in text sequences"""
        patterns_found = []
        
        # Common prefix/suffix analysis
        if len(texts) > 1:
            common_prefix = self._find_common_prefix(texts)
            common_suffix = self._find_common_suffix(texts)
            
            if common_prefix:
                patterns_found.append({
                    "type": "common_prefix",
                    "pattern": common_prefix,
                    "confidence": 0.8
                })
            
            if common_suffix:
                patterns_found.append({
                    "type": "common_suffix",
                    "pattern": common_suffix,
                    "confidence": 0.8
                })
        
        # Regex pattern discovery
        regex_patterns = self._discover_regex_patterns(texts)
        patterns_found.extend(regex_patterns)
        
        return {
            "texts": texts,
            "patterns_found": patterns_found,
            "pattern_count": len(patterns_found)
        }
    
    def _find_common_prefix(self, texts: List[str]) -> str:
        """Find common prefix in text list"""
        if not texts:
            return ""
        
        min_len = min(len(text) for text in texts)
        prefix = ""
        
        for i in range(min_len):
            char = texts[0][i]
            if all(text[i] == char for text in texts):
                prefix += char
            else:
                break
        
        return prefix
    
    def _find_common_suffix(self, texts: List[str]) -> str:
        """Find common suffix in text list"""
        if not texts:
            return ""
        
        reversed_texts = [text[::-1] for text in texts]
        reversed_prefix = self._find_common_prefix(reversed_texts)
        return reversed_prefix[::-1]
    
    def _discover_regex_patterns(self, texts: List[str]) -> List[Dict]:
        """Discover regex patterns in texts"""
        patterns = []
        
        # Check for numeric patterns
        if all(re.match(r'^\d+$', text) for text in texts):
            patterns.append({
                "type": "all_numeric",
                "pattern": r'^\d+$',
                "confidence": 1.0
            })
        
        # Check for email patterns
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if all(re.match(email_pattern, text) for text in texts):
            patterns.append({
                "type": "email_addresses",
                "pattern": email_pattern,
                "confidence": 1.0
            })
        
        return patterns


class InferenceTools:
    """Inference tools for integration with ToolExecutor"""
    
    def __init__(self):
        self.logic_engine = LogicalInferenceEngine()
        self.pattern_engine = PatternRecognitionEngine()
    
    def logical_add_rule(self, args: List[str]) -> str:
        """Add logical rule: logical_add_rule(premise, conclusion, confidence)"""
        if len(args) < 2:
            return "Error: logical_add_rule requires at least 2 arguments (premise, conclusion)"
        
        premise = args[0]
        conclusion = args[1]
        confidence = float(args[2]) if len(args) > 2 else 1.0
        
        try:
            rule_id = self.logic_engine.add_rule(premise, conclusion, confidence)
            return f"Added logical rule {rule_id}: '{premise}' → '{conclusion}' (confidence: {confidence})"
        except Exception as e:
            return f"Error adding rule: {str(e)}"
    
    def logical_add_fact(self, args: List[str]) -> str:
        """Add logical fact: logical_add_fact(fact, confidence)"""
        if len(args) < 1:
            return "Error: logical_add_fact requires 1 argument (fact)"
        
        fact = args[0]
        confidence = float(args[1]) if len(args) > 1 else 1.0
        
        try:
            fact_id = self.logic_engine.add_fact(fact, confidence)
            return f"Added fact {fact_id}: '{fact}' (confidence: {confidence})"
        except Exception as e:
            return f"Error adding fact: {str(e)}"
    
    def logical_infer(self, args: List[str]) -> str:
        """Perform logical inference: logical_infer(method)"""
        method = args[0] if args else "forward"
        
        try:
            if method == "forward":
                new_facts = self.logic_engine.forward_chaining()
                if new_facts:
                    result = f"Forward chaining derived {len(new_facts)} new facts:\n"
                    for fact in new_facts:
                        result += f"  • {fact['statement']} (confidence: {fact['confidence']:.2f})\n"
                    return result
                else:
                    return "Forward chaining completed, no new facts derived"
            
            elif method.startswith("prove:"):
                goal = method.split(":", 1)[1]
                proof = self.logic_engine.backward_chaining(goal)
                result = f"Backward chaining for goal: '{goal}'\n"
                result += f"Proved: {proof['proved']}\n"
                result += f"Proof steps: {proof['steps_count']}\n"
                return result
            
            else:
                return f"Error: Unknown inference method '{method}'"
            
        except Exception as e:
            return f"Error in logical inference: {str(e)}"
    
    def pattern_analyze_numeric(self, args: List[str]) -> str:
        """Analyze numeric patterns: pattern_analyze_numeric(num1, num2, num3, ...)"""
        if len(args) < 3:
            return "Error: pattern_analyze_numeric requires at least 3 numbers"
        
        try:
            sequence = [float(arg) for arg in args]
            analysis = self.pattern_engine.find_numeric_patterns(sequence)
            
            if "error" in analysis:
                return f"Pattern analysis error: {analysis['error']}"
            
            patterns = analysis["patterns_found"]
            if not patterns:
                return f"No clear patterns found in sequence: {sequence}"
            
            result = f"Pattern analysis for sequence {sequence}:\n"
            for pattern in patterns:
                result += f"  • {pattern['type']}: {pattern.get('common_difference', pattern.get('common_ratio', 'detected'))} "
                result += f"(confidence: {pattern['confidence']:.1f})\n"
            
            return result
            
        except ValueError:
            return "Error: All arguments must be valid numbers"
        except Exception as e:
            return f"Error in pattern analysis: {str(e)}"
    
    def pattern_analyze_text(self, args: List[str]) -> str:
        """Analyze text patterns: pattern_analyze_text(text1, text2, text3, ...)"""
        if len(args) < 2:
            return "Error: pattern_analyze_text requires at least 2 text strings"
        
        try:
            analysis = self.pattern_engine.find_text_patterns(args)
            
            patterns = analysis["patterns_found"]
            if not patterns:
                return f"No clear patterns found in texts: {args}"
            
            result = f"Text pattern analysis for {len(args)} texts:\n"
            for pattern in patterns:
                result += f"  • {pattern['type']}: '{pattern.get('pattern', 'detected')}' "
                result += f"(confidence: {pattern['confidence']:.1f})\n"
            
            return result
            
        except Exception as e:
            return f"Error in text pattern analysis: {str(e)}"
    
    def inference_status(self, args: List[str]) -> str:
        """Get inference engine status: inference_status()"""
        try:
            consistency = self.logic_engine.analyze_consistency()
            
            result = "Inference Engine Status:\n"
            result += f"Facts in knowledge base: {consistency['total_facts']}\n"
            result += f"Rules in knowledge base: {consistency['total_rules']}\n"
            result += f"Knowledge base consistent: {consistency['consistent']}\n"
            
            if consistency['contradictions']:
                result += f"Contradictions found: {len(consistency['contradictions'])}\n"
            
            result += f"Pattern sequences stored: {len(self.pattern_engine.data_sequences)}\n"
            
            return result
            
        except Exception as e:
            return f"Error getting inference status: {str(e)}"


# Integration helper
def get_inference_tools() -> Dict[str, callable]:
    """Get inference tools for ToolExecutor integration"""
    inference_tools = InferenceTools()
    
    return {
        "logical_add_rule": inference_tools.logical_add_rule,
        "logical_add_fact": inference_tools.logical_add_fact,
        "logical_infer": inference_tools.logical_infer,
        "pattern_analyze_numeric": inference_tools.pattern_analyze_numeric,
        "pattern_analyze_text": inference_tools.pattern_analyze_text,
        "inference_status": inference_tools.inference_status
    }
