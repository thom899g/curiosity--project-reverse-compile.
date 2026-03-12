"""
AST-based static analysis pipeline for OpenClaw orchestrator.
Architectural Choice: Two-phase analysis (AST parsing + NetworkX graph conversion)
enables both detailed code inspection and relationship analysis for the
knowledge graph.
"""
import ast
import logging
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import hashlib

import pandas as pd
import networkx as nx
from networkx import DiGraph

from firebase_setup import get_firestore_client

logger = logging.getLogger(__name__)

@dataclass
class CodeEntity:
    """Unified representation of code elements for Firestore storage."""
    entity_id: str
    entity_type: str  # 'function', 'class', 'module', 'variable', 'import'
    name: str
    file_path: str
    start_line: int
    end_line: int
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    ast_hash: str = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.ast_hash is None and self.file_path and Path(self.file_path).exists():
            self.ast_hash = self._compute_ast_hash()
    
    def _compute_ast_hash(self) -> str:
        """Compute hash of relevant AST content for change detection."""
        try:
            content = Path(self.file_path).read_text()
            lines = content.split('\n')[self.start_line-1:self.end_line]
            relevant_content = f"{self.entity_type}:{self.name}:" + ''.join(lines)
            return hashlib.md5(relevant_content.encode()).hexdigest()
        except Exception:
            return ""

@dataclass
class ControlFlowEdge:
    """Representation of control flow relationships."""
    source_id: str
    target_id: str
    edge_type: str  # 'calls', 'inherits', 'contains', 'imports'
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

class StaticAnalyzer:
    """AST-based static analyzer with Firestore integration."""
    
    def __init__(self, firestore_client=None):
        self.firestore_client = firestore_client or get_firestore_client()
        self.code_entities: Dict[str, CodeEntity] = {}
        self.control_flow_graph = DiGraph()
        self._entity_counter = 0
        
    def analyze_file(self, file_path: str) -> Tuple[Dict[str, CodeEntity], DiGraph]:
        """
        Analyze a single Python file, extracting code entities and relationships.
        
        Args:
            file_path: Path to Python file to analyze
            
        Returns:
            Tuple of (code_entities_dict, control_flow_graph)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            SyntaxError: If file has syntax errors
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Analyzing file: {file_path}")
        
        try:
            # Parse AST
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content, filename=str(file_path))
            
            # Reset state for new file
            file_entities = {}
            
            # Walk AST and extract entities
            for node in ast.walk(tree):
                entities = self._extract_entities_from_node(node, str(file_path))
                for entity in entities:
                    file_entities[entity.entity_id] = entity
                    self.code_entities[entity.entity_id] = entity
                    self.control_flow_graph.add_node(
                        entity.entity_id,
                        **asdict(entity)
                    )
            
            # Extract relationships
            for node in ast.walk(tree):
                self._extract_relationships_from_node(node, file_entities)
            
            logger.info(f"Extracted {len(file_entities)} entities from {file_path}")
            return file_entities, self.control_flow_graph
            
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            raise
    
    def analyze_directory(self, directory_path: str) -> Tuple[Dict[str, CodeEntity], DiGraph]:
        """
        Recursively analyze all Python files in directory.
        
        Args:
            directory_path: Root directory to analyze
            
        Returns:
            Tuple of (all_code_entities, combined_control_flow_graph)
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        all_entities = {}
        
        # Find all Python files
        python_files = list(directory.rglob("*.py"))
        logger.info(f"Found {len(python_files)} Python files in {directory}")
        
        for py_file in python_files:
            try:
                file_entities, _ = self.analyze_file(str(py_file))
                all_entities.update(file_entities)
            except Exception as e:
                logger.warning(f"Failed to analyze {py_file}: {e}")
                continue
        
        return all_entities, self.control_flow_graph
    
    def _extract_entities_from_node(self, node: ast.AST, file_path: str) -> List[CodeEntity]:
        """Extract CodeEntity objects from AST node."""
        entities = []
        
        if isinstance(node, ast.FunctionDef):
            entity = CodeEntity(
                entity_id=f"func_{node.name}_{node.lineno}_{self._entity_counter}",
                entity_type="function",
                name=node.name,
                file_path=file_path,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                metadata={
                    "args": [arg.arg for arg in node.args.args],
                    "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
                    "has_return": any(isinstance(n, ast.Return) for n in ast.walk(node)),
                    "has_yield": any(isinstance(n, ast.Yield) for n in ast.walk(node))
                }
            )
            entities.append(entity)
            self._entity_counter += 1
            
        elif isinstance(node, ast.ClassDef):
            entity = CodeEntity(
                entity_id=f"class_{node.name}_{node.lineno}_{self._entity_counter}",
                entity_type="class",
                name=node.name,
                file_path=file_path,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                metadata={
                    "bases": [self._get_base_name(base) for base in node.bases],
                    "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
                    "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                }
            )
            entities.append(entity)
            self._entity_counter += 1
            
        elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            for alias in node.names:
                entity = CodeEntity(
                    entity_id=f"import_{alias.name}_{self._entity_counter}",
                    entity_type="import",
                    name=alias.name,
                    file_path=file_path,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    metadata={
                        "module": node.module if isinstance(node, ast.ImportFrom) else None,
                        "alias": alias.asname,
                        "is_from": isinstance(node, ast.ImportFrom)
                    }
                )
                entities.append(entity)
                self._entity_counter += 1
        
        return entities
    
    def _extract_relationships_from_node(self, node: ast.AST, file_entities: Dict[str, CodeEntity]):
        """Extract control flow relationships from AST node."""
        if isinstance(node, ast.FunctionDef):
            func_id = f"func_{node.name}_{node.lineno}"
            for child in ast.walk(node):
                # Find function calls
                if isinstance(child, ast.Call):
                    func_name = self._get_call_name(child)
                    if func_name:
                        for target_id, target_entity in file_entities.items():
                            if (target_entity.entity_type == "function" and 
                                target_entity.name == func_name):
                                self.control_flow_graph.add_edge(
                                    func_id, target_id,
                                    edge_type='calls',
                                    context={'line': child.lineno}
                                )
        
        elif isinstance(node, ast.ClassDef):
            class_id = f"class_{node.name}_{node.lineno}"
            # Inheritance relationships
            for base in node.bases:
                base_name = self._get_base_name(base)
                for target_id, target_entity in file_entities.items():
                    if (target_entity.entity_type == "class" and 
                        target_entity.name == base_name):
                        self.control_flow_graph.add_edge(
                            class_id, target_id,
                            edge_type='inherits',
                            context={'line': node.lineno}
                        )
    
    def _get_decorator_name(self, node: ast.AST) -> str:
        """Extract decorator name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Call):
            return self._get_call_name(node)
        return str(node)
    
    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Extract function name from Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.f