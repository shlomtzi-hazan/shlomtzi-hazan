"""
Evaluation module for assessing the quality of AI assistant responses.

This module provides tools and metrics to evaluate how well the AI's responses
match expected "golden" outputs across various dimensions.
"""

import json
import re
import numpy as np
from typing import Dict, List, Tuple, Any
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer
import difflib

# Initialize NLP tools
try:
    nlp = spacy.load("en_core_web_md")
except:
    # Fallback if the model isn't installed
    import subprocess
    subprocess.call(['python', '-m', 'spacy', 'download', 'en_core_web_md'])
    nlp = spacy.load("en_core_web_md")


class ResponseEvaluator:
    """Evaluates AI assistant responses against golden (expected) outputs."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def evaluate_response(self, actual_response: str, golden_response: str) -> Dict[str, float]:
        """
        Evaluate a response across multiple dimensions.
        
        Args:
            actual_response: The AI's actual response
            golden_response: The expected "golden" response
            
        Returns:
            Dictionary of evaluation metrics with scores
        """
        results = {}
        
        # Content similarity metrics
        results["semantic_similarity"] = self.semantic_similarity(actual_response, golden_response)
        results["factual_overlap"] = self.factual_overlap(actual_response, golden_response)
        results["code_similarity"] = self.code_block_similarity(actual_response, golden_response)
        
        # Calculate overall score (weighted average)
        weights = {
            "semantic_similarity": 0.5,
            "factual_overlap": 0.25,
            "code_similarity": 0.25
        }
        
        weighted_sum = sum(score * weights[metric] for metric, score in results.items())
        total_weight = sum(weights.values())
        results["overall_score"] = weighted_sum / total_weight
        
        return results
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using SpaCy embeddings."""
        if not text1 or not text2:
            return 0.0
            
        doc1 = nlp(text1)
        doc2 = nlp(text2)
        
        if doc1.vector_norm == 0 or doc2.vector_norm == 0:
            return 0.0
            
        return doc1.similarity(doc2)
    
    def factual_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate factual overlap between texts based on key information extraction.
        
        This measures how well the response captures key entities, facts, and technical details
        from the golden response.
        """
        if not text1 or not text2:
            return 0.0
            
        # Extract entities from both texts
        doc1 = nlp(text1)
        doc2 = nlp(text2)
        
        # Get named entities, technical terms, and numbers
        entities1 = {e.text.lower() for e in doc1.ents}
        entities2 = {e.text.lower() for e in doc2.ents}
        
        # Add important technical terms (code keywords, function names, etc.)
        tech_pattern = r'\b(?:function|class|def|import|const|var|let|return|async|await|if|else|for|while|try|catch|except)\b'
        tech_terms1 = set(re.findall(tech_pattern, text1.lower()))
        tech_terms2 = set(re.findall(tech_pattern, text2.lower()))
        
        # Extract code-like identifiers (variable names, function names) 
        identifier_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        identifiers1 = set(re.findall(identifier_pattern, text1))
        identifiers2 = set(re.findall(identifier_pattern, text2))
        
        # Calculate overlap scores
        entities_overlap = self._calculate_overlap(entities1, entities2)
        tech_terms_overlap = self._calculate_overlap(tech_terms1, tech_terms2)
        identifiers_overlap = self._calculate_overlap(identifiers1, identifiers2)
        
        # Weighted combination of overlaps
        return 0.4 * entities_overlap + 0.3 * tech_terms_overlap + 0.3 * identifiers_overlap
    
    def _calculate_overlap(self, set1: set, set2: set) -> float:
        """Calculate overlap between two sets using Jaccard similarity."""
        if not set1 or not set2:
            return 0.0
            
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def code_block_similarity(self, text1: str, text2: str) -> float:
        """
        Evaluate similarity between code blocks in the responses.
        
        Extracts code blocks and compares them using specialized metrics.
        """
        code_blocks1 = self._extract_code_blocks(text1)
        code_blocks2 = self._extract_code_blocks(text2)
        
        if not code_blocks1 or not code_blocks2:
            # If one lacks code blocks but the other has them, penalize
            if code_blocks1 or code_blocks2:
                return 0.0
            # If neither has code blocks, consider it neutral (0.5)
            return 0.5
        
        # Compare all combinations of code blocks and take the best average match
        similarities = []
        for block1 in code_blocks1:
            block_scores = []
            for block2 in code_blocks2:
                # Use a combination of token similarity and sequence matching
                token_sim = self._token_similarity(block1, block2)
                seq_match = self._sequence_matcher_similarity(block1, block2)
                block_scores.append(0.6 * token_sim + 0.4 * seq_match)
            
            # Take the best match for this code block
            if block_scores:
                similarities.append(max(block_scores))
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from markdown-style content."""
        # Match code blocks with triple backticks
        pattern = r'```(?:\w+)?\s*(.*?)```'
        blocks = re.findall(pattern, text, re.DOTALL)
        
        # Also match indented code blocks
        indented_pattern = r'(?:^|\n)( {4}|\t)(.+?)(?=\n\S|\Z)'
        indented_blocks = re.findall(indented_pattern, text, re.DOTALL)
        indented_blocks = [''.join(block[1]) for block in indented_blocks if block[1].strip()]
        
        return blocks + indented_blocks
    
    def _token_similarity(self, text1: str, text2: str) -> float:
        """Compare code blocks based on token frequency."""
        if not text1 or not text2:
            return 0.0
            
        # Transform texts to TF-IDF vectors
        try:
            vectors = self.vectorizer.fit_transform([text1, text2])
            return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        except:
            return 0.0
    
    def _sequence_matcher_similarity(self, text1: str, text2: str) -> float:
        """Compare code using Python's sequence matcher for structural similarity."""
        if not text1 or not text2:
            return 0.0
            
        matcher = difflib.SequenceMatcher(None, text1, text2)
        return matcher.ratio()
    

class TestRunner:
    """Run tests against the AI assistant using predefined test cases."""
    
    def __init__(self, test_cases_file: str, assistant):
        """
        Initialize the test runner.
        
        Args:
            test_cases_file: Path to the JSON file containing test cases
            assistant: The OakieAssistant instance to test
        """
        self.test_cases = self._load_test_cases(test_cases_file)
        self.assistant = assistant
        self.evaluator = ResponseEvaluator()
        self.results = {}
        
    def _load_test_cases(self, file_path: str) -> List[Dict[str, Any]]:
        """Load test cases from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data.get('test_cases', [])
        except Exception as e:
            print(f"Error loading test cases: {e}")
            return []
    
    def run_tests(self, threshold: float = 0.7) -> Dict[str, Any]:
        """
        Run all test cases and evaluate the results.
        
        Args:
            threshold: Minimum overall score to pass a test
            
        Returns:
            Dictionary with test results
        """
        overall_results = {
            "total_tests": len(self.test_cases),
            "passed_tests": 0,
            "average_score": 0.0,
            "test_details": []
        }
        
        total_score = 0.0
        
        for test_case in self.test_cases:
            result = self.run_single_test(test_case, threshold)
            overall_results["test_details"].append(result)
            
            if result["passed"]:
                overall_results["passed_tests"] += 1
                
            total_score += result["scores"]["overall_score"]
        
        if self.test_cases:
            overall_results["average_score"] = total_score / len(self.test_cases)
            
        return overall_results
    
    def run_single_test(self, test_case: Dict[str, Any], threshold: float) -> Dict[str, Any]:
        """
        Run a single test case and evaluate the result.
        
        Args:
            test_case: The test case to run
            threshold: Minimum overall score to pass the test
            
        Returns:
            Dictionary with test result
        """
        test_input = test_case["input"]
        golden_output = test_case["golden_output"]["content"]
        
        # Prepare documents if provided
        documents = []
        for doc_name in test_input.get("documents", []):
            try:
                # Resolve the absolute path of the document
                doc_path = os.path.join(os.path.dirname(__file__), doc_name)
                # Try to load the document
                self.assistant.load_document_from_file(doc_path)
                documents.append(doc_name)
            except Exception as e:
                print(f"Warning: Could not load document {doc_name}: {e}")
        
        # Get response from assistant
        actual_output = self.assistant.ask(
            test_input["query"],
            include_documents=True,
            mentions=test_input.get("mentions", [])
        )
        
        # Evaluate response
        scores = self.evaluator.evaluate_response(actual_output, golden_output)
        
        # Determine if test passed
        passed = scores["overall_score"] >= threshold
        
        return {
            "id": test_case["id"],
            "name": test_case["name"],
            "input": test_input,
            "golden_output": golden_output,
            "actual_output": actual_output,
            "scores": scores,
            "passed": passed
        }
    
    def generate_report(self, results: Dict[str, Any], output_file: str = None) -> str:
        """
        Generate a report from the test results.
        
        Args:
            results: Results dictionary from run_tests
            output_file: Optional file to write the report to
            
        Returns:
            Report as a string
        """
        report = []
        
        # Add summary
        report.append("# Oakie AI Assistant Test Results")
        report.append(f"Test Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append(f"## Summary")
        if results["total_tests"] > 0:
            passed_percentage = results["passed_tests"] / results["total_tests"] * 100
        else:
            passed_percentage = 0.0
        report.append(f"- **Tests Passed**: {results['passed_tests']}/{results['total_tests']} ({passed_percentage:.1f}%)")
        report.append(f"- **Average Score**: {results['average_score']:.3f}")
        report.append("")
        
        # Add detailed results for each test
        report.append("## Detailed Results")
        for test_result in results["test_details"]:
            report.append("")
            report.append(f"### {test_result['name']} ({test_result['id']})")
            report.append(f"**Status**: {'PASSED' if test_result['passed'] else 'FAILED'}")
            report.append("")
            report.append("#### Scores")
            report.append("| Metric | Score |")
            report.append("| ------ | ----- |")
            for metric, score in test_result["scores"].items():
                report.append(f"| {metric} | {score:.3f} |")
            
            report.append("")
            report.append("#### Query")
            report.append(f"```\n{test_result['input']['query']}\n```")
            
            if test_result["scores"]["overall_score"] < 0.6:
                report.append("")
                report.append("#### Comparison")
                report.append("*Golden Output:*")
                report.append("```")
                report.append(test_result["golden_output"][:300] + "..." if len(test_result["golden_output"]) > 300 else test_result["golden_output"])
                report.append("```")
                
                report.append("*Actual Output:*")
                report.append("```")
                report.append(test_result["actual_output"][:300] + "..." if len(test_result["actual_output"]) > 300 else test_result["actual_output"])
                report.append("```")
        
        report_text = "\n".join(report)
        
        # Write to file if specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_text)
            except Exception as e:
                print(f"Error writing report to file: {e}")
        
        return report_text


class PromptEvaluation:
    def __init__(self, guidelines_path: str):
        self.guidelines = self._load_guidelines(guidelines_path)

    def _load_guidelines(self, path: str) -> Dict[str, Any]:
        """Load company guidelines from a JSON file."""
        try:
            with open(path, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load guidelines: {e}")

    def evaluate_response(self, response: str, expected: str) -> float:
        """Evaluate the response based on guidelines and expected output.
        
        Args:
            response: The AI assistant's response.
            expected: The expected correct response.
        
        Returns:
            A score between 0 and 1 indicating the quality of the response.
        """
        score = 0.0
        if response.strip() == expected.strip():
            score = 1.0
        else:
            # Implement more sophisticated evaluation logic here
            score = self._compare_responses(response, expected)
        return score

    def _compare_responses(self, response: str, expected: str) -> float:
        """Compare the response with the expected output.
        
        Args:
            response: The AI assistant's response.
            expected: The expected correct response.
        
        Returns:
            A score between 0 and 1 indicating the similarity.
        """
        # Placeholder for actual comparison logic
        # This could involve semantic similarity, keyword matching, etc.
        return 0.5  # Example placeholder score

    def evaluate_responses(self, responses: List[Dict[str, str]]) -> List[float]:
        """Evaluate multiple responses.
        
        Args:
            responses: A list of dictionaries with 'response' and 'expected' keys.
        
        Returns:
            A list of scores for each response.
        """
        return [self.evaluate_response(item['response'], item['expected']) for item in responses]

# Example usage
if __name__ == "__main__":
    import sys
    import datetime
    import os
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from main import OakieAssistant
    
    print("Initializing Oakie Assistant for evaluation...")
    assistant = OakieAssistant()
    
    test_cases_file = os.path.join(os.path.dirname(__file__), "test_cases.json")
    if not os.path.exists(test_cases_file):
        print(f"Error: Test cases file not found at {test_cases_file}")
        sys.exit(1)
    
    runner = TestRunner(test_cases_file, assistant)
    
    print(f"Running {len(runner.test_cases)} test cases...")
    results = runner.run_tests()
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"test_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    
    report = runner.generate_report(results, output_file)
    
    print("\nEvaluation Results:")
    print(f"- Tests Passed: {results['passed_tests']}/{results['total_tests']} ({results['passed_tests']/results['total_tests']*100:.1f}%)")
    print(f"- Average Score: {results['average_score']:.3f}")
    print(f"- Detailed report written to: {output_file}")
    
    if results['average_score'] < 0.7:
        print("\nWARNING: Overall score is below threshold (0.7). Consider improving prompts or test cases.")
        sys.exit(1)
