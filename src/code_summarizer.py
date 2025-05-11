"""
Code summarization module using LLM models
"""
import os
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import re

class CodeSummarizer:
    def __init__(self, model_name: str = "Salesforce/codet5-base"):
        """
        Initialize code summarizer with CodeT5 model
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        print(f"\nInitializing CodeT5 model from {model_name}...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Loading model...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        print("Model initialization complete!")
        
        # Set maximum lengths for input and output
        self.max_input_length = 1024  # Increased for better context
        self.max_output_length = 256  # Increased for more detailed summaries
        
    def extract_class_info(self, code: str) -> Dict[str, any]:
        """Extract detailed class information from Java code"""
        class_info = {
            'name': '',
            'package': '',
            'description': '',
            'extends': '',
            'implements': [],
            'methods': [],
            'fields': [],
            'imports': []
        }
        
        try:
            lines = code.split('\n')
            in_comment = False
            comment_buffer = []
            
            for line in lines:
                line = line.strip()
                
                # Handle multi-line comments
                if line.startswith('/**') or line.startswith('/*'):
                    in_comment = True
                    continue
                    
                if in_comment:
                    if line.endswith('*/'):
                        in_comment = False
                        class_info['description'] = ' '.join(comment_buffer)
                        comment_buffer = []
                    elif line.startswith('*'):
                        comment_buffer.append(line.lstrip('* '))
                    continue
                
                # Extract package
                if line.startswith('package '):
                    class_info['package'] = line[8:].rstrip(';')
                    
                # Extract imports
                elif line.startswith('import '):
                    class_info['imports'].append(line[7:].rstrip(';'))
                    
                # Extract class declaration
                elif 'class ' in line:
                    parts = line.split('class ')[1].split()
                    class_info['name'] = parts[0]
                    
                    # Extract inheritance
                    if 'extends' in line:
                        class_info['extends'] = line.split('extends')[1].split('{')[0].strip()
                    
                    # Extract interfaces
                    if 'implements' in line:
                        implements = line.split('implements')[1].split('{')[0].strip()
                        class_info['implements'] = [i.strip() for i in implements.split(',')]
                
                # Extract methods with parameters
                elif ('public' in line or 'private' in line or 'protected' in line) and '(' in line and ')' in line:
                    method_match = re.match(r'(?:public|private|protected)\s+(?:static\s+)?(\w+)\s+(\w+)\s*\((.*?)\)', line)
                    if method_match:
                        return_type, name, params = method_match.groups()
                        class_info['methods'].append({
                            'name': name,
                            'return_type': return_type,
                            'parameters': [p.strip() for p in params.split(',') if p.strip()]
                        })
                
                # Extract fields
                elif ('public' in line or 'private' in line or 'protected' in line) and ';' in line and '(' not in line:
                    field_match = re.match(r'(?:public|private|protected)\s+(?:static\s+)?(\w+)\s+(\w+)\s*;', line)
                    if field_match:
                        type_name, field_name = field_match.groups()
                        class_info['fields'].append({
                            'name': field_name,
                            'type': type_name
                        })
                        
            return class_info
        except Exception as e:
            print(f"Warning: Error extracting class info: {str(e)}")
            return class_info
            
    def generate_class_summary(self, code: str) -> str:
        """Generate a comprehensive summary for a Java class"""
        try:
            # Extract class information
            class_info = self.extract_class_info(code)
            
            # Build structured input text
            sections = []
            
            # Add class header
            header = f"Class: {class_info['name']}"
            if class_info['package']:
                header += f"\nPackage: {class_info['package']}"
            if class_info['extends']:
                header += f"\nExtends: {class_info['extends']}"
            if class_info['implements']:
                header += f"\nImplements: {', '.join(class_info['implements'])}"
            sections.append(header)
            
            # Add description if available
            if class_info['description']:
                sections.append(f"Description:\n{class_info['description']}")
            
            # Add methods summary
            if class_info['methods']:
                method_lines = []
                for method in class_info['methods']:
                    params = ', '.join(method['parameters']) if method['parameters'] else 'void'
                    method_lines.append(f"- {method['name']}({params}) -> {method['return_type']}")
                sections.append("Methods:\n" + '\n'.join(method_lines))
            
            # Add fields summary
            if class_info['fields']:
                field_lines = [f"- {field['name']}: {field['type']}" for field in class_info['fields']]
                sections.append("Fields:\n" + '\n'.join(field_lines))
            
            # Add dependencies
            if class_info['imports']:
                sections.append("Dependencies:\n" + '\n'.join(f"- {imp}" for imp in class_info['imports']))
            
            # Combine all sections
            input_text = '\n\n'.join(sections)
            
            # Generate summary using model
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_output_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2
            )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Format the final summary
            final_summary = f"{summary}\n\n"
            if class_info['package']:
                final_summary += f"Package: {class_info['package']}\n"
            if class_info['extends']:
                final_summary += f"Extends: {class_info['extends']}\n"
            if class_info['implements']:
                final_summary += f"Implements: {', '.join(class_info['implements'])}\n"
            if class_info['imports']:
                final_summary += f"Dependencies: {', '.join(class_info['imports'])}"
            
            return final_summary.strip()
            
        except Exception as e:
            print(f"Warning: Failed to generate class summary: {str(e)}")
            return "Error generating summary"
    
    def summarize_code(self, code: str) -> str:
        """Generate a summary for code"""
        try:
            print("  - Extracting class information...")
            class_info = self.extract_class_info(code)
            print(f"  - Class info extracted: {class_info['name'] if class_info['name'] else 'Unknown class'}")
            
            if not class_info['name']:
                print("  - Warning: Could not extract class name")
            
            print("  - Generating class summary...")
            summary = self.generate_class_summary(code)
            print(f"  - Summary generated: {len(summary)} characters")
            return summary
        except Exception as e:
            print(f"  - Error in summarize_code: {str(e)}")
            print("  - Returning error message as summary")
            return "Error generating summary"
    
    def summarize_text(self, text: str) -> str:
        """Summarize natural language text (for requirements)"""
        try:
            print("  - Preparing input text...")
            # Extract key information from requirement
            requirement_info = self._extract_requirement_info(text)
            
            # Format requirement with extracted info
            formatted_text = f"""
ORIGINAL REQUIREMENT:
{text}

KEY INFORMATION:
- Actions: {', '.join(requirement_info['actions'])}
- Actors: {', '.join(requirement_info['actors'])}
- Objects: {', '.join(requirement_info['objects'])}
- Constraints: {', '.join(requirement_info['constraints'])}
"""
            print("  - Tokenizing input...")
            inputs = self.tokenizer(
                formatted_text,
                max_length=self.max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            print("  - Generating summary with model...")
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_output_length,
                num_beams=4,
                temperature=0.3,  # Even lower temperature for more focused output
                top_p=0.95,
                length_penalty=1.0,
                repetition_penalty=1.5,
                early_stopping=True,
                do_sample=False
            )
            
            print("  - Decoding output tokens...")
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Combine original text with generated summary
            final_text = f"""
REQUIREMENT TEXT:
{text}

EXTRACTED INFORMATION:
{summary}
"""
            print(f"  - Final text generated: {len(final_text)} characters")
            return final_text.strip()
        except Exception as e:
            print(f"  - Error in summarize_text: {str(e)}")
            return text.strip()

    def _extract_requirement_info(self, text: str) -> Dict[str, List[str]]:
        """Extract key information from requirement text"""
        info = {
            'actions': [],
            'actors': [],
            'objects': [],
            'constraints': []
        }
        
        # Common action words in requirements
        action_words = {'shall', 'must', 'will', 'should', 'can', 'may', 'allow', 'enable', 'provide', 'support'}
        
        # Common actor words
        actor_words = {'system', 'user', 'admin', 'administrator', 'patient', 'doctor', 'nurse', 'HCP', 'LHCP', 'UHCP'}
        
        # Split text into sentences
        sentences = text.split('.')
        
        for sentence in sentences:
            words = sentence.strip().lower().split()
            
            # Extract actions
            for i, word in enumerate(words):
                if word in action_words:
                    # Get the verb phrase
                    verb_phrase = []
                    for j in range(i+1, min(i+4, len(words))):
                        if words[j] in {'to', 'the', 'a', 'an'}:
                            continue
                        verb_phrase.append(words[j])
                    if verb_phrase:
                        info['actions'].append(' '.join(verb_phrase))
            
            # Extract actors
            for word in words:
                if word in actor_words:
                    info['actors'].append(word)
            
            # Extract objects (nouns after verbs)
            for i, word in enumerate(words[:-1]):
                if word in action_words and i < len(words)-1:
                    obj_phrase = []
                    for j in range(i+1, min(i+5, len(words))):
                        if words[j] not in {'to', 'the', 'a', 'an'}:
                            obj_phrase.append(words[j])
                    if obj_phrase:
                        info['objects'].append(' '.join(obj_phrase))
            
            # Extract constraints (conditions and qualifiers)
            constraint_markers = {'if', 'when', 'unless', 'only', 'after', 'before', 'during', 'while'}
            for i, word in enumerate(words):
                if word in constraint_markers:
                    constraint = []
                    for j in range(i, min(i+8, len(words))):
                        constraint.append(words[j])
                    if constraint:
                        info['constraints'].append(' '.join(constraint))
        
        # Remove duplicates and empty strings
        for key in info:
            info[key] = list(set(filter(None, info[key])))
        
        return info