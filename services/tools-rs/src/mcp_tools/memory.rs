use super::Tool;
use crate::mcp_core::{McpResult, McpTool};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

/// Entity in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    #[serde(rename = "entityType")]
    pub entity_type: String,
    pub observations: Vec<String>,
}

/// Relation between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub from: String,
    pub to: String,
    #[serde(rename = "relationType")]
    pub relation_type: String,
}

/// Knowledge graph containing entities and relations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    pub entities: Vec<Entity>,
    pub relations: Vec<Relation>,
}

/// Storage format for entities (includes type field)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StorageEntity {
    #[serde(rename = "type")]
    storage_type: String,
    name: String,
    #[serde(rename = "entityType")]
    entity_type: String,
    observations: Vec<String>,
}

/// Storage format for relations (includes type field)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StorageRelation {
    #[serde(rename = "type")]
    storage_type: String,
    from: String,
    to: String,
    #[serde(rename = "relationType")]
    relation_type: String,
}

/// Parameters for creating entities
#[derive(Debug, Deserialize)]
struct CreateEntitiesParams {
    entities: Vec<Entity>,
}

/// Parameters for creating relations
#[derive(Debug, Deserialize)]
struct CreateRelationsParams {
    relations: Vec<Relation>,
}

/// Parameters for adding observations
#[derive(Debug, Deserialize)]
struct AddObservationsParams {
    observations: Vec<ObservationInput>,
}

/// Input for adding observations
#[derive(Debug, Deserialize)]
struct ObservationInput {
    #[serde(rename = "entityName")]
    entity_name: String,
    contents: Vec<String>,
}

/// Parameters for deleting entities
#[derive(Debug, Deserialize)]
struct DeleteEntitiesParams {
    #[serde(rename = "entityNames")]
    entity_names: Vec<String>,
}

/// Parameters for deleting observations
#[derive(Debug, Deserialize)]
struct DeleteObservationsParams {
    deletions: Vec<ObservationDeletion>,
}

/// Observation deletion specification
#[derive(Debug, Deserialize)]
struct ObservationDeletion {
    #[serde(rename = "entityName")]
    entity_name: String,
    observations: Vec<String>,
}

/// Parameters for deleting relations
#[derive(Debug, Deserialize)]
struct DeleteRelationsParams {
    relations: Vec<Relation>,
}

/// Parameters for searching nodes
#[derive(Debug, Deserialize)]
struct SearchNodesParams {
    query: String,
}

/// Parameters for opening specific nodes
#[derive(Debug, Deserialize)]
struct OpenNodesParams {
    names: Vec<String>,
}

/// Memory management tool for knowledge graphs
pub struct MemoryManagementTool {
    memory_file_path: Mutex<String>,
}

impl Default for MemoryManagementTool {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryManagementTool {
    pub fn new() -> Self {
        // Đảm bảo luôn lấy lại MEMORY_FILE_PATH mới nhất cho mỗi instance
        let current_dir = env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        let default_memory_path = current_dir.join("memory.json");

        let memory_file_path = match env::var("MEMORY_FILE_PATH") {
            Ok(path) => {
                if Path::new(&path).is_absolute() {
                    path
                } else {
                    current_dir.join(path).to_string_lossy().to_string()
                }
            }
            Err(_) => default_memory_path.to_string_lossy().to_string(),
        };

        // Luôn tạo file nếu chưa tồn tại để tránh lỗi read_to_string
        if !Path::new(&memory_file_path).exists() {
            let _ = fs::File::create(&memory_file_path);
        }

        Self {
            memory_file_path: Mutex::new(memory_file_path),
        }
    }

    /// Load knowledge graph from file
    fn load_graph(&self) -> Result<KnowledgeGraph, Box<dyn std::error::Error>> {
        let path = self.memory_file_path.lock().unwrap().clone();

        match fs::read_to_string(&path) {
            Ok(data) => {
                let lines: Vec<&str> = data
                    .lines()
                    .filter(|line| !line.trim().is_empty())
                    .collect();
                let mut entities = Vec::new();
                let mut relations = Vec::new();

                for line in lines {
                    if let Ok(value) = serde_json::from_str::<Value>(line) {
                        if let Some(item_type) = value.get("type").and_then(|v| v.as_str()) {
                            if item_type == "entity" {
                                if let Ok(storage_entity) =
                                    serde_json::from_value::<StorageEntity>(value)
                                {
                                    entities.push(Entity {
                                        name: storage_entity.name,
                                        entity_type: storage_entity.entity_type,
                                        observations: storage_entity.observations,
                                    });
                                }
                            } else if item_type == "relation" {
                                if let Ok(storage_relation) =
                                    serde_json::from_value::<StorageRelation>(value)
                                {
                                    relations.push(Relation {
                                        from: storage_relation.from,
                                        to: storage_relation.to,
                                        relation_type: storage_relation.relation_type,
                                    });
                                }
                            }
                        }
                    }
                }

                Ok(KnowledgeGraph {
                    entities,
                    relations,
                })
            }
            Err(_) => Ok(KnowledgeGraph {
                entities: Vec::new(),
                relations: Vec::new(),
            }),
        }
    }

    /// Save knowledge graph to file
    fn save_graph(&self, graph: &KnowledgeGraph) -> Result<(), Box<dyn std::error::Error>> {
        let path = self.memory_file_path.lock().unwrap().clone();

        let mut lines = Vec::new();

        // Save entities
        for entity in &graph.entities {
            let storage_entity = StorageEntity {
                storage_type: "entity".to_string(),
                name: entity.name.clone(),
                entity_type: entity.entity_type.clone(),
                observations: entity.observations.clone(),
            };
            lines.push(serde_json::to_string(&storage_entity)?);
        }

        // Save relations
        for relation in &graph.relations {
            let storage_relation = StorageRelation {
                storage_type: "relation".to_string(),
                from: relation.from.clone(),
                to: relation.to.clone(),
                relation_type: relation.relation_type.clone(),
            };
            lines.push(serde_json::to_string(&storage_relation)?);
        }

        fs::write(&path, lines.join("\n"))?;
        Ok(())
    }

    /// Create new entities
    fn create_entities(
        &self,
        entities: Vec<Entity>,
    ) -> Result<Vec<Entity>, Box<dyn std::error::Error>> {
        let mut graph = self.load_graph()?;
        let mut new_entities = Vec::new();

        for entity in entities {
            // Check if entity already exists
            if !graph.entities.iter().any(|e| e.name == entity.name) {
                graph.entities.push(entity.clone());
                new_entities.push(entity);
            }
        }

        self.save_graph(&graph)?;
        Ok(new_entities)
    }

    /// Create new relations
    fn create_relations(
        &self,
        relations: Vec<Relation>,
    ) -> Result<Vec<Relation>, Box<dyn std::error::Error>> {
        let mut graph = self.load_graph()?;
        let mut new_relations = Vec::new();

        for relation in relations {
            // Check if relation already exists
            if !graph.relations.iter().any(|r| {
                r.from == relation.from
                    && r.to == relation.to
                    && r.relation_type == relation.relation_type
            }) {
                graph.relations.push(relation.clone());
                new_relations.push(relation);
            }
        }

        self.save_graph(&graph)?;
        Ok(new_relations)
    }

    /// Add observations to entities
    fn add_observations(
        &self,
        observations: Vec<ObservationInput>,
    ) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
        let mut graph = self.load_graph()?;
        let mut results = Vec::new();

        for obs in observations {
            if let Some(entity) = graph
                .entities
                .iter_mut()
                .find(|e| e.name == obs.entity_name)
            {
                let mut added_observations = Vec::new();
                for content in obs.contents {
                    if !entity.observations.contains(&content) {
                        entity.observations.push(content.clone());
                        added_observations.push(content);
                    }
                }
                results.push(json!({
                    "entityName": obs.entity_name,
                    "addedObservations": added_observations
                }));
            } else {
                return Err(format!("Entity with name {} not found", obs.entity_name).into());
            }
        }

        self.save_graph(&graph)?;
        Ok(results)
    }

    /// Delete entities and their relations
    fn delete_entities(&self, entity_names: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
        let mut graph = self.load_graph()?;

        // Remove entities
        graph.entities.retain(|e| !entity_names.contains(&e.name));

        // Remove relations involving deleted entities
        graph
            .relations
            .retain(|r| !entity_names.contains(&r.from) && !entity_names.contains(&r.to));

        self.save_graph(&graph)?;
        Ok(())
    }

    /// Delete specific observations from entities
    fn delete_observations(
        &self,
        deletions: Vec<ObservationDeletion>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut graph = self.load_graph()?;

        for deletion in deletions {
            if let Some(entity) = graph
                .entities
                .iter_mut()
                .find(|e| e.name == deletion.entity_name)
            {
                entity
                    .observations
                    .retain(|o| !deletion.observations.contains(o));
            }
        }

        self.save_graph(&graph)?;
        Ok(())
    }

    /// Delete relations
    fn delete_relations(&self, relations: Vec<Relation>) -> Result<(), Box<dyn std::error::Error>> {
        let mut graph = self.load_graph()?;

        graph.relations.retain(|r| {
            !relations.iter().any(|del_rel| {
                r.from == del_rel.from
                    && r.to == del_rel.to
                    && r.relation_type == del_rel.relation_type
            })
        });

        self.save_graph(&graph)?;
        Ok(())
    }

    /// Read entire graph
    fn read_graph(&self) -> Result<KnowledgeGraph, Box<dyn std::error::Error>> {
        self.load_graph()
    }

    /// Search nodes based on query
    fn search_nodes(&self, query: String) -> Result<KnowledgeGraph, Box<dyn std::error::Error>> {
        let graph = self.load_graph()?;
        let query_lower = query.to_lowercase();

        // Filter entities
        let filtered_entities: Vec<Entity> = graph
            .entities
            .into_iter()
            .filter(|e| {
                e.name.to_lowercase().contains(&query_lower)
                    || e.entity_type.to_lowercase().contains(&query_lower)
                    || e.observations
                        .iter()
                        .any(|o| o.to_lowercase().contains(&query_lower))
            })
            .collect();

        // Get entity names for filtering relations
        let entity_names: HashSet<String> =
            filtered_entities.iter().map(|e| e.name.clone()).collect();

        // Filter relations
        let filtered_relations: Vec<Relation> = graph
            .relations
            .into_iter()
            .filter(|r| entity_names.contains(&r.from) && entity_names.contains(&r.to))
            .collect();

        Ok(KnowledgeGraph {
            entities: filtered_entities,
            relations: filtered_relations,
        })
    }

    /// Open specific nodes by names
    fn open_nodes(&self, names: Vec<String>) -> Result<KnowledgeGraph, Box<dyn std::error::Error>> {
        let graph = self.load_graph()?;

        // Filter entities
        let filtered_entities: Vec<Entity> = graph
            .entities
            .into_iter()
            .filter(|e| names.contains(&e.name))
            .collect();

        // Get entity names for filtering relations
        let entity_names: HashSet<String> =
            filtered_entities.iter().map(|e| e.name.clone()).collect();

        // Filter relations
        let filtered_relations: Vec<Relation> = graph
            .relations
            .into_iter()
            .filter(|r| entity_names.contains(&r.from) && entity_names.contains(&r.to))
            .collect();

        Ok(KnowledgeGraph {
            entities: filtered_entities,
            relations: filtered_relations,
        })
    }
}

impl Tool for MemoryManagementTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "memory_management".to_string(),
            description: "A comprehensive memory management tool for knowledge graphs with entities and relations".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "create_entities",
                            "create_relations", 
                            "add_observations",
                            "delete_entities",
                            "delete_observations",
                            "delete_relations",
                            "read_graph",
                            "search_nodes",
                            "open_nodes"
                        ],
                        "description": "The memory operation to perform"
                    },
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string", "description": "The name of the entity" },
                                "entityType": { "type": "string", "description": "The type of the entity" },
                                "observations": { 
                                    "type": "array", 
                                    "items": { "type": "string" },
                                    "description": "An array of observation contents associated with the entity"
                                },
                            },
                            "required": ["name", "entityType", "observations"]
                        },
                        "description": "Array of entities for create_entities operation"
                    },
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from": { "type": "string", "description": "The name of the entity where the relation starts" },
                                "to": { "type": "string", "description": "The name of the entity where the relation ends" },
                                "relationType": { "type": "string", "description": "The type of the relation" },
                            },
                            "required": ["from", "to", "relationType"]
                        },
                        "description": "Array of relations for create_relations or delete_relations operations"
                    },
                    "observations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entityName": { "type": "string", "description": "The name of the entity to add the observations to" },
                                "contents": { 
                                    "type": "array", 
                                    "items": { "type": "string" },
                                    "description": "An array of observation contents to add"
                                },
                            },
                            "required": ["entityName", "contents"]
                        },
                        "description": "Array of observations for add_observations operation"
                    },
                    "entityNames": { 
                        "type": "array", 
                        "items": { "type": "string" },
                        "description": "An array of entity names for delete_entities operation" 
                    },
                    "deletions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entityName": { "type": "string", "description": "The name of the entity containing the observations" },
                                "observations": { 
                                    "type": "array", 
                                    "items": { "type": "string" },
                                    "description": "An array of observations to delete"
                                },
                            },
                            "required": ["entityName", "observations"]
                        },
                        "description": "Array of observation deletions for delete_observations operation"
                    },
                    "query": { 
                        "type": "string", 
                        "description": "The search query to match against entity names, types, and observation content for search_nodes operation" 
                    },
                    "names": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "An array of entity names to retrieve for open_nodes operation",
                    }
                },
                "required": ["operation"]
            }),
        }
    }

    fn execute(&self, params: Value) -> McpResult<Value> {
        let operation = params
            .get("operation")
            .and_then(|v| v.as_str())
            .ok_or("Missing or invalid operation parameter")?;

        let result = match operation {
            "create_entities" => {
                let entities_params: CreateEntitiesParams = serde_json::from_value(params)?;
                let new_entities = self.create_entities(entities_params.entities)?;
                json!(new_entities)
            }
            "create_relations" => {
                let relations_params: CreateRelationsParams = serde_json::from_value(params)?;
                let new_relations = self.create_relations(relations_params.relations)?;
                json!(new_relations)
            }
            "add_observations" => {
                let obs_params: AddObservationsParams = serde_json::from_value(params)?;
                let results = self.add_observations(obs_params.observations)?;
                json!(results)
            }
            "delete_entities" => {
                let delete_params: DeleteEntitiesParams = serde_json::from_value(params)?;
                self.delete_entities(delete_params.entity_names)?;
                json!("Entities deleted successfully")
            }
            "delete_observations" => {
                let delete_params: DeleteObservationsParams = serde_json::from_value(params)?;
                self.delete_observations(delete_params.deletions)?;
                json!("Observations deleted successfully")
            }
            "delete_relations" => {
                let delete_params: DeleteRelationsParams = serde_json::from_value(params)?;
                self.delete_relations(delete_params.relations)?;
                json!("Relations deleted successfully")
            }
            "read_graph" => {
                let graph = self.read_graph()?;
                json!(graph)
            }
            "search_nodes" => {
                let search_params: SearchNodesParams = serde_json::from_value(params)?;
                let result_graph = self.search_nodes(search_params.query)?;
                json!(result_graph)
            }
            "open_nodes" => {
                let open_params: OpenNodesParams = serde_json::from_value(params)?;
                let result_graph = self.open_nodes(open_params.names)?;
                json!(result_graph)
            }
            _ => return Err(format!("Unknown operation: {}", operation).into()),
        };

        Ok(json!({
            "content": [{
                "type": "text",
                "text": serde_json::to_string_pretty(&result)?
            }]
        }))
    }
}

// ============== Individual Memory Tools ==============

/// Individual tool for creating entities
pub struct MemoryCreateEntitesTool {
    memory_tool: MemoryManagementTool,
}

impl MemoryCreateEntitesTool {
    pub fn new() -> Self {
        Self {
            memory_tool: MemoryManagementTool::new(),
        }
    }
}

impl Default for MemoryCreateEntitesTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for MemoryCreateEntitesTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "memory_create_entities".to_string(),
            description: "Create multiple new entities in the knowledge graph".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string", "description": "The name of the entity" },
                                "entityType": { "type": "string", "description": "The type of the entity" },
                                "observations": {
                                    "type": "array",
                                    "items": { "type": "string" },
                                    "description": "An array of observation contents associated with the entity"
                                },
                            },
                            "required": ["name", "entityType", "observations"]
                        },
                    },
                },
                "required": ["entities"]
            }),
        }
    }

    fn execute(&self, params: Value) -> McpResult<Value> {
        let mut params_with_op = params;
        params_with_op["operation"] = json!("create_entities");
        self.memory_tool.execute(params_with_op)
    }
}

/// Individual tool for creating relations
pub struct MemoryCreateRelationsTool {
    memory_tool: MemoryManagementTool,
}

impl MemoryCreateRelationsTool {
    pub fn new() -> Self {
        Self {
            memory_tool: MemoryManagementTool::new(),
        }
    }
}

impl Default for MemoryCreateRelationsTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for MemoryCreateRelationsTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "memory_create_relations".to_string(),
            description: "Create multiple new relations between entities in the knowledge graph. Relations should be in active voice".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from": { "type": "string", "description": "The name of the entity where the relation starts" },
                                "to": { "type": "string", "description": "The name of the entity where the relation ends" },
                                "relationType": { "type": "string", "description": "The type of the relation" },
                            },
                            "required": ["from", "to", "relationType"]
                        },
                    },
                },
                "required": ["relations"]
            }),
        }
    }

    fn execute(&self, params: Value) -> McpResult<Value> {
        let mut params_with_op = params;
        params_with_op["operation"] = json!("create_relations");
        self.memory_tool.execute(params_with_op)
    }
}

/// Individual tool for adding observations
pub struct MemoryAddObservationsTool {
    memory_tool: MemoryManagementTool,
}

impl MemoryAddObservationsTool {
    pub fn new() -> Self {
        Self {
            memory_tool: MemoryManagementTool::new(),
        }
    }
}

impl Default for MemoryAddObservationsTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for MemoryAddObservationsTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "memory_add_observations".to_string(),
            description: "Add new observations to existing entities in the knowledge graph"
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "observations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entityName": { "type": "string", "description": "The name of the entity to add the observations to" },
                                "contents": {
                                    "type": "array",
                                    "items": { "type": "string" },
                                    "description": "An array of observation contents to add"
                                },
                            },
                            "required": ["entityName", "contents"]
                        },
                    },
                },
                "required": ["observations"]
            }),
        }
    }

    fn execute(&self, params: Value) -> McpResult<Value> {
        let mut params_with_op = params;
        params_with_op["operation"] = json!("add_observations");
        self.memory_tool.execute(params_with_op)
    }
}

/// Individual tool for deleting entities
pub struct MemoryDeleteEntitiesTool {
    memory_tool: MemoryManagementTool,
}

impl MemoryDeleteEntitiesTool {
    pub fn new() -> Self {
        Self {
            memory_tool: MemoryManagementTool::new(),
        }
    }
}

impl Default for MemoryDeleteEntitiesTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for MemoryDeleteEntitiesTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "memory_delete_entities".to_string(),
            description:
                "Delete multiple entities and their associated relations from the knowledge graph"
                    .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "entityNames": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "An array of entity names to delete"
                    },
                },
                "required": ["entityNames"]
            }),
        }
    }

    fn execute(&self, params: Value) -> McpResult<Value> {
        let mut params_with_op = params;
        params_with_op["operation"] = json!("delete_entities");
        self.memory_tool.execute(params_with_op)
    }
}

/// Individual tool for deleting observations
pub struct MemoryDeleteObservationsTool {
    memory_tool: MemoryManagementTool,
}

impl MemoryDeleteObservationsTool {
    pub fn new() -> Self {
        Self {
            memory_tool: MemoryManagementTool::new(),
        }
    }
}

impl Default for MemoryDeleteObservationsTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for MemoryDeleteObservationsTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "memory_delete_observations".to_string(),
            description: "Delete specific observations from entities in the knowledge graph"
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "deletions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entityName": { "type": "string", "description": "The name of the entity containing the observations" },
                                "observations": {
                                    "type": "array",
                                    "items": { "type": "string" },
                                    "description": "An array of observations to delete"
                                },
                            },
                            "required": ["entityName", "observations"]
                        },
                    },
                },
                "required": ["deletions"]
            }),
        }
    }

    fn execute(&self, params: Value) -> McpResult<Value> {
        let mut params_with_op = params;
        params_with_op["operation"] = json!("delete_observations");
        self.memory_tool.execute(params_with_op)
    }
}

/// Individual tool for deleting relations
pub struct MemoryDeleteRelationsTool {
    memory_tool: MemoryManagementTool,
}

impl MemoryDeleteRelationsTool {
    pub fn new() -> Self {
        Self {
            memory_tool: MemoryManagementTool::new(),
        }
    }
}

impl Default for MemoryDeleteRelationsTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for MemoryDeleteRelationsTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "memory_delete_relations".to_string(),
            description: "Delete multiple relations from the knowledge graph".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from": { "type": "string", "description": "The name of the entity where the relation starts" },
                                "to": { "type": "string", "description": "The name of the entity where the relation ends" },
                                "relationType": { "type": "string", "description": "The type of the relation" },
                            },
                            "required": ["from", "to", "relationType"]
                        },
                        "description": "An array of relations to delete"
                    },
                },
                "required": ["relations"]
            }),
        }
    }

    fn execute(&self, params: Value) -> McpResult<Value> {
        let mut params_with_op = params;
        params_with_op["operation"] = json!("delete_relations");
        self.memory_tool.execute(params_with_op)
    }
}

/// Individual tool for reading the entire graph
pub struct MemoryReadGraphTool {
    memory_tool: MemoryManagementTool,
}

impl MemoryReadGraphTool {
    pub fn new() -> Self {
        Self {
            memory_tool: MemoryManagementTool::new(),
        }
    }
}

impl Default for MemoryReadGraphTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for MemoryReadGraphTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "memory_read_graph".to_string(),
            description: "Read the entire knowledge graph".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {},
            }),
        }
    }

    fn execute(&self, params: Value) -> McpResult<Value> {
        let mut params_with_op = params;
        params_with_op["operation"] = json!("read_graph");
        self.memory_tool.execute(params_with_op)
    }
}

/// Individual tool for searching nodes
pub struct MemorySearchNodesTool {
    memory_tool: MemoryManagementTool,
}

impl MemorySearchNodesTool {
    pub fn new() -> Self {
        Self {
            memory_tool: MemoryManagementTool::new(),
        }
    }
}

impl Default for MemorySearchNodesTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for MemorySearchNodesTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "memory_search_nodes".to_string(),
            description: "Search for nodes in the knowledge graph based on a query".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "The search query to match against entity names, types, and observation content" },
                },
                "required": ["query"]
            }),
        }
    }

    fn execute(&self, params: Value) -> McpResult<Value> {
        let mut params_with_op = params;
        params_with_op["operation"] = json!("search_nodes");
        self.memory_tool.execute(params_with_op)
    }
}

/// Individual tool for opening specific nodes
pub struct MemoryOpenNodesTool {
    memory_tool: MemoryManagementTool,
}

impl MemoryOpenNodesTool {
    pub fn new() -> Self {
        Self {
            memory_tool: MemoryManagementTool::new(),
        }
    }
}

impl Default for MemoryOpenNodesTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for MemoryOpenNodesTool {
    fn definition(&self) -> McpTool {
        McpTool {
            name: "memory_open_nodes".to_string(),
            description: "Open specific nodes in the knowledge graph by their names".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "An array of entity names to retrieve",
                    },
                },
                "required": ["names"]
            }),
        }
    }

    fn execute(&self, params: Value) -> McpResult<Value> {
        let mut params_with_op = params;
        params_with_op["operation"] = json!("open_nodes");
        self.memory_tool.execute(params_with_op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_memory_tool_definition() {
        let tool = MemoryManagementTool::new();
        let def = tool.definition();
        assert_eq!(def.name, "memory_management");
    }

    #[test]
    fn test_create_and_read_entities() {
        let temp_file = NamedTempFile::new().unwrap();
        std::env::set_var("MEMORY_FILE_PATH", temp_file.path());

        let tool = MemoryManagementTool::new();

        let entities = vec![Entity {
            name: "test_entity".to_string(),
            entity_type: "test_type".to_string(),
            observations: vec!["observation1".to_string()],
        }];

        let result = tool.create_entities(entities).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "test_entity");

        let graph = tool.read_graph().unwrap();
        assert_eq!(graph.entities.len(), 1);
        assert_eq!(graph.entities[0].name, "test_entity");
    }

    #[test]
    fn test_create_relations() {
        let temp_file = NamedTempFile::new().unwrap();
        std::env::set_var("MEMORY_FILE_PATH", temp_file.path());

        let tool = MemoryManagementTool::new();

        let relations = vec![Relation {
            from: "entity1".to_string(),
            to: "entity2".to_string(),
            relation_type: "relates_to".to_string(),
        }];

        let result = tool.create_relations(relations).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].from, "entity1");

        let graph = tool.read_graph().unwrap();
        assert_eq!(graph.relations.len(), 1);
        assert_eq!(graph.relations[0].from, "entity1");
    }
}
