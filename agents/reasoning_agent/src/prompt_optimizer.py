"""
Prompt Optimizer for Dynamic Prompt Engineering
Implements intelligent prompt generation, optimization, and A/B testing
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from datetime import datetime
import hashlib
import statistics


class PromptType(Enum):
    """Types of prompts for optimization"""

    REASONING = "reasoning"
    INSTRUCTION = "instruction"
    QUESTION = "question"
    COMPLETION = "completion"
    CLASSIFICATION = "classification"
    GENERATION = "generation"


class OptimizationStrategy(Enum):
    """Prompt optimization strategies"""

    TEMPLATE_BASED = "template_based"
    EXAMPLE_DRIVEN = "example_driven"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    STEP_BY_STEP = "step_by_step"
    ROLE_PLAYING = "role_playing"
    CONTEXT_INJECTION = "context_injection"


@dataclass
class PromptTemplate:
    """Template for prompt generation"""

    id: str = ""
    name: str = ""
    prompt_type: PromptType = PromptType.REASONING
    template: str = ""
    variables: List[str] = field(default_factory=list)
    examples: List[Dict] = field(default_factory=list)
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.TEMPLATE_BASED
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PromptVariant:
    """Individual prompt variant for A/B testing"""

    id: str = ""
    template_id: str = ""
    content: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_score: float = 0.0
    execution_count: int = 0
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    avg_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptOptimizer:
    """
    Intelligent prompt optimizer for dynamic prompt engineering

    Features:
    1. Dynamic prompt generation based on context
    2. Template-based prompt optimization
    3. A/B testing for prompt variants
    4. Performance-based prompt selection
    5. Context-aware prompt adaptation
    6. Multi-strategy prompt optimization
    """

    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self.variants: Dict[str, List[PromptVariant]] = {}
        self.performance_history: List[Dict] = []
        self.optimization_rules: List[Dict] = []

        # Initialize with default templates
        self._initialize_default_templates()

    async def optimize_prompt(
        self,
        prompt_type: PromptType,
        context: Dict[str, Any],
        target_task: str,
        optimization_goals: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Main prompt optimization method

        Args:
            prompt_type: Type of prompt to optimize
            context: Context for prompt generation
            target_task: Specific task the prompt should accomplish
            optimization_goals: Goals for optimization (e.g., "accuracy", "speed", "clarity")

        Returns:
            Optimized prompt with performance metrics
        """
        print(f"🎯 Optimizing {prompt_type.value} prompt for: {target_task[:50]}...")

        start_time = time.time()
        optimization_goals = optimization_goals or ["accuracy", "clarity"]

        # Step 1: Generate candidate prompts
        candidates = await self._generate_candidate_prompts(
            prompt_type, context, target_task
        )

        # Step 2: Apply optimization strategies
        optimized_candidates = await self._apply_optimization_strategies(
            candidates, optimization_goals
        )

        # Step 3: A/B test variants
        test_results = await self._ab_test_variants(optimized_candidates, context)

        # Step 4: Select best performing prompt
        best_prompt = await self._select_best_prompt(test_results, optimization_goals)

        # Step 5: Apply final optimizations
        final_prompt = await self._apply_final_optimizations(best_prompt, context)

        optimization_time = time.time() - start_time

        result = {
            "optimized_prompt": final_prompt["content"],
            "prompt_metadata": final_prompt,
            "optimization_summary": {
                "candidates_generated": len(candidates),
                "strategies_applied": len(optimized_candidates),
                "variants_tested": len(test_results),
                "optimization_time": optimization_time,
                "performance_improvement": final_prompt.get("improvement_score", 0),
            },
            "ab_test_results": test_results,
            "recommendations": await self._generate_optimization_recommendations(
                final_prompt
            ),
        }

        print(f"✅ Prompt optimization completed in {optimization_time:.2f}s")
        print(f"   • Performance score: {final_prompt.get('performance_score', 0):.2f}")
        print(f"   • Candidates tested: {len(candidates)}")

        return result

    def _initialize_default_templates(self):
        """Initialize default prompt templates"""

        # Reasoning template
        reasoning_template = PromptTemplate(
            id="reasoning_basic",
            name="Basic Reasoning Template",
            prompt_type=PromptType.REASONING,
            template="""You are an expert reasoning assistant. 

Task: {task}
Context: {context}

Please think through this step by step:
1. Analyze the problem
2. Consider relevant factors
3. Apply logical reasoning
4. Provide your conclusion

Reasoning: """,
            variables=["task", "context"],
            optimization_strategy=OptimizationStrategy.STEP_BY_STEP,
        )
        self.templates[reasoning_template.id] = reasoning_template

        # Chain of thought template
        cot_template = PromptTemplate(
            id="chain_of_thought",
            name="Chain of Thought Template",
            prompt_type=PromptType.REASONING,
            template="""Let's work through this step by step.

Problem: {problem}

Step 1: What do we know?
{context}

Step 2: What do we need to find?
{goal}

Step 3: What's our approach?
Let me think through this carefully...

Step 4: Solution
""",
            variables=["problem", "context", "goal"],
            optimization_strategy=OptimizationStrategy.CHAIN_OF_THOUGHT,
        )
        self.templates[cot_template.id] = cot_template

        # Role-playing template
        role_template = PromptTemplate(
            id="expert_role",
            name="Expert Role Template",
            prompt_type=PromptType.INSTRUCTION,
            template="""You are a {role} with {experience} years of experience in {domain}.

Your task: {task}

Given your expertise, please:
1. Apply your specialized knowledge
2. Consider best practices in {domain}
3. Provide actionable recommendations
4. Explain your reasoning

Response: """,
            variables=["role", "experience", "domain", "task"],
            optimization_strategy=OptimizationStrategy.ROLE_PLAYING,
        )
        self.templates[role_template.id] = role_template

    async def _generate_candidate_prompts(
        self, prompt_type: PromptType, context: Dict[str, Any], target_task: str
    ) -> List[Dict[str, Any]]:
        """Generate candidate prompts for optimization"""

        candidates = []

        # Find relevant templates
        relevant_templates = [
            template
            for template in self.templates.values()
            if template.prompt_type == prompt_type
        ]

        for template in relevant_templates:
            # Generate variants from template
            template_variants = await self._generate_template_variants(
                template, context, target_task
            )
            candidates.extend(template_variants)

        # Generate custom prompts based on context
        custom_prompts = await self._generate_custom_prompts(
            prompt_type, context, target_task
        )
        candidates.extend(custom_prompts)

        return candidates

    async def _generate_template_variants(
        self, template: PromptTemplate, context: Dict[str, Any], target_task: str
    ) -> List[Dict[str, Any]]:
        """Generate variants from a template"""

        variants = []

        # Basic template filling
        basic_variant = await self._fill_template_basic(template, context, target_task)
        variants.append(basic_variant)

        # Enhanced template with additional context
        enhanced_variant = await self._fill_template_enhanced(
            template, context, target_task
        )
        variants.append(enhanced_variant)

        # Simplified template
        simplified_variant = await self._fill_template_simplified(
            template, context, target_task
        )
        variants.append(simplified_variant)

        return variants

    async def _fill_template_basic(
        self, template: PromptTemplate, context: Dict[str, Any], target_task: str
    ) -> Dict[str, Any]:
        """Fill template with basic parameters"""

        # Map context to template variables
        template_params = {}
        for var in template.variables:
            if var == "task":
                template_params[var] = target_task
            elif var == "context":
                template_params[var] = str(context.get("description", ""))
            elif var == "problem":
                template_params[var] = target_task
            elif var == "goal":
                template_params[var] = context.get("goal", "solve the problem")
            elif var in context:
                template_params[var] = str(context[var])
            else:
                template_params[var] = f"[{var}]"  # Placeholder

        try:
            filled_content = template.template.format(**template_params)
        except KeyError as e:
            filled_content = template.template  # Fallback to original

        return {
            "content": filled_content,
            "template_id": template.id,
            "variant_type": "basic",
            "parameters": template_params,
            "strategy": template.optimization_strategy.value,
        }

    async def _fill_template_enhanced(
        self, template: PromptTemplate, context: Dict[str, Any], target_task: str
    ) -> Dict[str, Any]:
        """Fill template with enhanced context and examples"""

        basic_variant = await self._fill_template_basic(template, context, target_task)

        # Add examples if available
        examples_text = ""
        if template.examples:
            examples_text = "\n\nExamples:\n"
            for i, example in enumerate(template.examples[:2]):  # Limit to 2 examples
                examples_text += f"Example {i+1}: {example.get('input', '')} → {example.get('output', '')}\n"

        # Add additional context
        context_text = ""
        if context.get("additional_info"):
            context_text = f"\n\nAdditional Context:\n{context['additional_info']}"

        enhanced_content = basic_variant["content"] + examples_text + context_text

        return {
            **basic_variant,
            "content": enhanced_content,
            "variant_type": "enhanced",
            "enhancements": ["examples", "additional_context"],
        }

    async def _fill_template_simplified(
        self, template: PromptTemplate, context: Dict[str, Any], target_task: str
    ) -> Dict[str, Any]:
        """Fill template with simplified, concise approach"""

        # Create simplified version
        simplified_template = re.sub(
            r"\n\d+\..*?\n", "\n", template.template
        )  # Remove numbered steps
        simplified_template = re.sub(
            r"Please.*?:", "", simplified_template
        )  # Remove verbose instructions
        simplified_template = simplified_template.replace(
            "\n\n", "\n"
        )  # Reduce spacing

        # Use simplified template
        template_params = {
            "task": target_task,
            "context": str(context.get("description", "")),
        }

        try:
            filled_content = simplified_template.format(**template_params)
        except KeyError:
            filled_content = f"Task: {target_task}\nContext: {context.get('description', '')}\nResponse:"

        return {
            "content": filled_content,
            "template_id": template.id,
            "variant_type": "simplified",
            "parameters": template_params,
            "strategy": "simplified",
        }

    async def _generate_custom_prompts(
        self, prompt_type: PromptType, context: Dict[str, Any], target_task: str
    ) -> List[Dict[str, Any]]:
        """Generate custom prompts based on context analysis"""

        custom_prompts = []

        # Direct prompt
        direct_prompt = {
            "content": f"{target_task}\n\nPlease provide a detailed response:",
            "template_id": "custom_direct",
            "variant_type": "direct",
            "strategy": "direct",
        }
        custom_prompts.append(direct_prompt)

        # Question-based prompt
        if prompt_type == PromptType.QUESTION:
            question_prompt = {
                "content": f"Question: {target_task}\n\nTo answer this thoroughly, I need to consider:\n1. Key factors\n2. Available evidence\n3. Logical implications\n\nAnswer:",
                "template_id": "custom_question",
                "variant_type": "analytical",
                "strategy": "analytical",
            }
            custom_prompts.append(question_prompt)

        # Context-rich prompt
        if context:
            context_prompt = {
                "content": f"Given the context: {context.get('description', '')}\n\nTask: {target_task}\n\nConsidering the context, my approach is:",
                "template_id": "custom_context",
                "variant_type": "context_rich",
                "strategy": "context_injection",
            }
            custom_prompts.append(context_prompt)

        return custom_prompts

    async def _apply_optimization_strategies(
        self, candidates: List[Dict[str, Any]], optimization_goals: List[str]
    ) -> List[Dict[str, Any]]:
        """Apply optimization strategies to candidate prompts"""

        optimized = []

        for candidate in candidates:
            # Apply goal-specific optimizations
            for goal in optimization_goals:
                optimized_variant = await self._apply_goal_optimization(candidate, goal)
                optimized.append(optimized_variant)

        return optimized

    async def _apply_goal_optimization(
        self, candidate: Dict[str, Any], goal: str
    ) -> Dict[str, Any]:
        """Apply optimization for specific goal"""

        content = candidate["content"]

        if goal == "accuracy":
            # Add precision instructions
            content += "\n\nPlease be precise and accurate in your response. Consider all relevant factors."
        elif goal == "clarity":
            # Add clarity instructions
            content += "\n\nPlease provide a clear, well-structured response that is easy to understand."
        elif goal == "speed":
            # Simplify for speed
            content = content.replace("\n\n", "\n")  # Reduce verbosity
            content += "\n\nPlease provide a concise response."
        elif goal == "creativity":
            # Add creativity prompts
            content += "\n\nPlease think creatively and consider innovative approaches."
        elif goal == "detail":
            # Add detail instructions
            content += "\n\nPlease provide a comprehensive, detailed response with examples where appropriate."

        optimized = candidate.copy()
        optimized["content"] = content
        optimized["optimization_goal"] = goal
        optimized["optimized"] = True

        return optimized

    async def _ab_test_variants(
        self, candidates: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Conduct A/B testing on prompt variants"""

        test_results = []

        for candidate in candidates:
            # Simulate A/B testing with mock performance metrics
            performance_metrics = await self._simulate_prompt_performance(
                candidate, context
            )

            test_result = {
                **candidate,
                "performance_metrics": performance_metrics,
                "test_id": hashlib.md5(candidate["content"].encode()).hexdigest()[:8],
            }

            test_results.append(test_result)

        return test_results

    async def _simulate_prompt_performance(
        self, candidate: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Simulate prompt performance for testing"""

        content = candidate["content"]

        # Mock performance calculation based on prompt characteristics
        base_score = 0.6

        # Length optimization
        length_factor = 1.0 - abs(len(content) - 200) / 1000  # Optimal around 200 chars

        # Structure bonus
        structure_bonus = (
            0.1 if any(marker in content for marker in ["Step", "1.", "2.", ":"]) else 0
        )

        # Clarity bonus
        clarity_bonus = (
            0.1
            if any(word in content.lower() for word in ["clear", "precise", "detailed"])
            else 0
        )

        # Context integration bonus
        context_bonus = (
            0.1 if context and any(str(v) in content for v in context.values()) else 0
        )

        # Strategy bonus
        strategy_bonus = {
            "chain_of_thought": 0.15,
            "step_by_step": 0.12,
            "role_playing": 0.10,
            "context_injection": 0.08,
        }.get(candidate.get("strategy", ""), 0)

        # Calculate final score
        performance_score = min(
            1.0,
            base_score
            + length_factor * 0.2
            + structure_bonus
            + clarity_bonus
            + context_bonus
            + strategy_bonus,
        )

        # Add some randomness to simulate real testing
        import random

        noise = random.uniform(-0.05, 0.05)
        performance_score = max(0.0, min(1.0, performance_score + noise))

        return {
            "overall_score": performance_score,
            "clarity_score": min(1.0, performance_score + clarity_bonus),
            "structure_score": min(1.0, base_score + structure_bonus + 0.1),
            "context_integration": min(1.0, base_score + context_bonus + 0.1),
            "estimated_response_time": random.uniform(1.0, 3.0),
            "estimated_accuracy": performance_score * 0.95,
            "engagement_score": random.uniform(0.6, 0.9),
        }

    async def _select_best_prompt(
        self, test_results: List[Dict[str, Any]], optimization_goals: List[str]
    ) -> Dict[str, Any]:
        """Select best performing prompt based on optimization goals"""

        if not test_results:
            return {"content": "Default prompt", "performance_score": 0.5}

        # Weight scores based on optimization goals
        goal_weights = {
            "accuracy": "estimated_accuracy",
            "clarity": "clarity_score",
            "speed": "estimated_response_time",  # Lower is better
            "detail": "structure_score",
            "creativity": "engagement_score",
        }

        best_prompt = None
        best_score = -1

        for result in test_results:
            metrics = result["performance_metrics"]
            weighted_score = 0

            for goal in optimization_goals:
                metric_key = goal_weights.get(goal, "overall_score")
                metric_value = metrics.get(metric_key, 0.5)

                # Invert for time-based metrics (lower is better)
                if "time" in metric_key:
                    metric_value = 1.0 / (1.0 + metric_value)

                weighted_score += metric_value

            # Average the weighted scores
            final_score = weighted_score / len(optimization_goals)

            if final_score > best_score:
                best_score = final_score
                best_prompt = result

        best_prompt["final_score"] = best_score
        return best_prompt

    async def _apply_final_optimizations(
        self, best_prompt: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply final optimizations to the selected prompt"""

        content = best_prompt["content"]

        # Remove redundant spacing
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Ensure proper formatting
        if not content.endswith((":", "?", ".")):
            content += ":"

        # Add context-specific enhancements
        if context.get("urgency") == "high":
            content = "URGENT: " + content

        if context.get("audience") == "technical":
            content += (
                "\n\nNote: Technical details and implementation specifics are welcome."
            )

        # Calculate improvement score
        original_score = 0.6  # Baseline
        current_score = best_prompt["performance_metrics"]["overall_score"]
        improvement_score = (current_score - original_score) / original_score

        final_prompt = best_prompt.copy()
        final_prompt["content"] = content
        final_prompt["improvement_score"] = improvement_score
        final_prompt["final_optimizations"] = [
            "spacing",
            "formatting",
            "context_specific",
        ]

        return final_prompt

    async def _generate_optimization_recommendations(
        self, final_prompt: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for further optimization"""

        recommendations = []

        performance = final_prompt["performance_metrics"]

        if performance["clarity_score"] < 0.8:
            recommendations.append(
                "Consider adding more explicit instructions for clarity"
            )

        if performance["structure_score"] < 0.7:
            recommendations.append(
                "Add numbered steps or bullet points for better structure"
            )

        if performance["context_integration"] < 0.7:
            recommendations.append("Integrate more context-specific information")

        if len(final_prompt["content"]) > 500:
            recommendations.append(
                "Consider shortening the prompt for better engagement"
            )

        if not any(
            word in final_prompt["content"].lower()
            for word in ["please", "step", "consider"]
        ):
            recommendations.append("Add polite language and guidance words")

        return recommendations

    async def dynamic_prompt_adaptation(
        self,
        base_prompt: str,
        execution_feedback: Dict[str, Any],
        context_changes: Dict[str, Any] = None,
    ) -> str:
        """Dynamically adapt prompt based on execution feedback"""

        adapted_prompt = base_prompt

        # Adapt based on success rate
        if execution_feedback.get("success_rate", 1.0) < 0.7:
            adapted_prompt += (
                "\n\nPlease be extra careful and thorough in your analysis."
            )

        # Adapt based on response quality
        if execution_feedback.get("avg_confidence", 1.0) < 0.8:
            adapted_prompt += "\n\nProvide your confidence level and reasoning."

        # Adapt based on context changes
        if context_changes:
            if context_changes.get("complexity_increased"):
                adapted_prompt += "\n\nNote: This is a complex problem requiring careful consideration."

            if context_changes.get("time_constraint"):
                adapted_prompt += "\n\nPlease provide a concise but complete response."

        return adapted_prompt

    def export_templates(self, filepath: str):
        """Export prompt templates to file"""
        export_data = {
            "templates": {
                tid: {
                    **template.__dict__,
                    "prompt_type": template.prompt_type.value,
                    "optimization_strategy": template.optimization_strategy.value,
                }
                for tid, template in self.templates.items()
            },
            "export_timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"💾 Prompt templates exported to: {filepath}")


# Helper functions for Prompt Optimizer tools integration
def get_prompt_optimizer_tools() -> Dict[str, Any]:
    """Get Prompt Optimizer tools for integration"""

    async def optimize_prompt_tool(params: Dict[str, Any]) -> Dict[str, Any]:
        """Tool wrapper for prompt optimization"""
        optimizer = PromptOptimizer()

        prompt_type = PromptType(params.get("prompt_type", "reasoning"))

        result = await optimizer.optimize_prompt(
            prompt_type=prompt_type,
            context=params.get("context", {}),
            target_task=params["task"],
            optimization_goals=params.get("goals", ["accuracy", "clarity"]),
        )

        return result

    async def adapt_prompt_dynamically(params: Dict[str, Any]) -> Dict[str, Any]:
        """Tool for dynamic prompt adaptation"""
        optimizer = PromptOptimizer()

        adapted = await optimizer.dynamic_prompt_adaptation(
            base_prompt=params["base_prompt"],
            execution_feedback=params.get("feedback", {}),
            context_changes=params.get("context_changes", {}),
        )

        return {
            "original_prompt": params["base_prompt"],
            "adapted_prompt": adapted,
            "adaptations_applied": len(adapted.split("\n"))
            - len(params["base_prompt"].split("\n")),
        }

    return {
        "optimize_prompt": {
            "function": optimize_prompt_tool,
            "description": "Optimize prompts for specific tasks and goals",
            "parameters": ["task", "prompt_type", "context", "goals"],
            "category": "optimization",
        },
        "adapt_prompt_dynamically": {
            "function": adapt_prompt_dynamically,
            "description": "Dynamically adapt prompts based on execution feedback",
            "parameters": ["base_prompt", "feedback", "context_changes"],
            "category": "adaptation",
        },
    }
