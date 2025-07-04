import { promises as fs } from 'fs';
import * as path from 'path';
import { Tool } from "@modelcontextprotocol/sdk/types.js";

// Helper to get current directory in both CommonJS and ESM
function getCurrentDir() {
  // @ts-ignore
  if (typeof __dirname !== 'undefined') return __dirname;
  return process.cwd();
}

// Memory interfaces
export interface Entity {
  name: string;
  entityType: string;
  observations: string[];
}

export interface Relation {
  from: string;
  to: string;
  relationType: string;
}

export interface KnowledgeGraph {
  entities: Entity[];
  relations: Relation[];
}

// Memory server class
export class MemoryServer {
  private memoryFilePath: string;

  constructor() {
    const currentDir = getCurrentDir();
    const defaultMemoryPath = path.join(currentDir, 'memory.json');
    
    this.memoryFilePath = process.env.MEMORY_FILE_PATH
      ? path.isAbsolute(process.env.MEMORY_FILE_PATH)
        ? process.env.MEMORY_FILE_PATH
        : path.join(currentDir, process.env.MEMORY_FILE_PATH)
      : defaultMemoryPath;
  }

  private async loadGraph(): Promise<KnowledgeGraph> {
    try {
      const data = await fs.readFile(this.memoryFilePath, "utf-8");
      const lines = data.split("\n").filter(line => line.trim() !== "");
      return lines.reduce((graph: KnowledgeGraph, line) => {
        const item = JSON.parse(line);
        if (item.type === "entity") graph.entities.push(item as Entity);
        if (item.type === "relation") graph.relations.push(item as Relation);
        return graph;
      }, { entities: [], relations: [] });
    } catch (error) {
      if (error instanceof Error && 'code' in error && (error as any).code === "ENOENT") {
        return { entities: [], relations: [] };
      }
      throw error;
    }
  }

  private async saveGraph(graph: KnowledgeGraph): Promise<void> {
    const lines = [
      ...graph.entities.map(e => JSON.stringify({ type: "entity", ...e })),
      ...graph.relations.map(r => JSON.stringify({ type: "relation", ...r })),
    ];
    await fs.writeFile(this.memoryFilePath, lines.join("\n"));
  }

  async createEntities(entities: Entity[]): Promise<Entity[]> {
    const graph = await this.loadGraph();
    const newEntities = entities.filter(e => !graph.entities.some(existingEntity => existingEntity.name === e.name));
    graph.entities.push(...newEntities);
    await this.saveGraph(graph);
    return newEntities;
  }

  async createRelations(relations: Relation[]): Promise<Relation[]> {
    const graph = await this.loadGraph();
    const newRelations = relations.filter(r => !graph.relations.some(existingRelation => 
      existingRelation.from === r.from && 
      existingRelation.to === r.to && 
      existingRelation.relationType === r.relationType
    ));
    graph.relations.push(...newRelations);
    await this.saveGraph(graph);
    return newRelations;
  }

  async addObservations(observations: { entityName: string; contents: string[] }[]): Promise<{ entityName: string; addedObservations: string[] }[]> {
    const graph = await this.loadGraph();
    const results = observations.map(o => {
      const entity = graph.entities.find(e => e.name === o.entityName);
      if (!entity) {
        throw new Error(`Entity with name ${o.entityName} not found`);
      }
      const newObservations = o.contents.filter(content => !entity.observations.includes(content));
      entity.observations.push(...newObservations);
      return { entityName: o.entityName, addedObservations: newObservations };
    });
    await this.saveGraph(graph);
    return results;
  }

  async deleteEntities(entityNames: string[]): Promise<void> {
    const graph = await this.loadGraph();
    graph.entities = graph.entities.filter(e => !entityNames.includes(e.name));
    graph.relations = graph.relations.filter(r => !entityNames.includes(r.from) && !entityNames.includes(r.to));
    await this.saveGraph(graph);
  }

  async deleteObservations(deletions: { entityName: string; observations: string[] }[]): Promise<void> {
    const graph = await this.loadGraph();
    deletions.forEach(d => {
      const entity = graph.entities.find(e => e.name === d.entityName);
      if (entity) {
        entity.observations = entity.observations.filter(o => !d.observations.includes(o));
      }
    });
    await this.saveGraph(graph);
  }

  async deleteRelations(relations: Relation[]): Promise<void> {
    const graph = await this.loadGraph();
    graph.relations = graph.relations.filter(r => !relations.some(delRelation => 
      r.from === delRelation.from && 
      r.to === delRelation.to && 
      r.relationType === delRelation.relationType
    ));
    await this.saveGraph(graph);
  }

  async readGraph(): Promise<KnowledgeGraph> {
    return this.loadGraph();
  }

  async searchNodes(query: string): Promise<KnowledgeGraph> {
    const graph = await this.loadGraph();
    
    const filteredEntities = graph.entities.filter(e => 
      e.name.toLowerCase().includes(query.toLowerCase()) ||
      e.entityType.toLowerCase().includes(query.toLowerCase()) ||
      e.observations.some(o => o.toLowerCase().includes(query.toLowerCase()))
    );
  
    const filteredEntityNames = new Set(filteredEntities.map(e => e.name));
  
    const filteredRelations = graph.relations.filter(r => 
      filteredEntityNames.has(r.from) && filteredEntityNames.has(r.to)
    );
  
    return {
      entities: filteredEntities,
      relations: filteredRelations,
    };
  }

  async openNodes(names: string[]): Promise<KnowledgeGraph> {
    const graph = await this.loadGraph();
    
    const filteredEntities = graph.entities.filter(e => names.includes(e.name));
    const filteredEntityNames = new Set(filteredEntities.map(e => e.name));
  
    const filteredRelations = graph.relations.filter(r => 
      filteredEntityNames.has(r.from) && filteredEntityNames.has(r.to)
    );
  
    return {
      entities: filteredEntities,
      relations: filteredRelations,
    };
  }

  // Memory tool call handlers
  public async processCreateEntities(input: unknown): Promise<{ content: Array<{ type: string; text: string }>; isError?: boolean }> {
    try {
      const args = input as { entities: Entity[] };
      const result = await this.createEntities(args.entities);
      return {
        content: [{
          type: "text",
          text: JSON.stringify({ created: result.length, entities: result }, null, 2)
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            error: error instanceof Error ? error.message : String(error),
            status: 'failed'
          }, null, 2)
        }],
        isError: true
      };
    }
  }

  public async processCreateRelations(input: unknown): Promise<{ content: Array<{ type: string; text: string }>; isError?: boolean }> {
    try {
      const args = input as { relations: Relation[] };
      const result = await this.createRelations(args.relations);
      return {
        content: [{
          type: "text",
          text: JSON.stringify({ created: result.length, relations: result }, null, 2)
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            error: error instanceof Error ? error.message : String(error),
            status: 'failed'
          }, null, 2)
        }],
        isError: true
      };
    }
  }

  public async processAddObservations(input: unknown): Promise<{ content: Array<{ type: string; text: string }>; isError?: boolean }> {
    try {
      const args = input as { observations: { entityName: string; contents: string[] }[] };
      const result = await this.addObservations(args.observations);
      return {
        content: [{
          type: "text",
          text: JSON.stringify(result, null, 2)
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            error: error instanceof Error ? error.message : String(error),
            status: 'failed'
          }, null, 2)
        }],
        isError: true
      };
    }
  }

  public async processDeleteEntities(input: unknown): Promise<{ content: Array<{ type: string; text: string }>; isError?: boolean }> {
    try {
      const args = input as { entityNames: string[] };
      await this.deleteEntities(args.entityNames);
      return {
        content: [{
          type: "text",
          text: JSON.stringify({ message: "Entities deleted successfully", deleted: args.entityNames }, null, 2)
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            error: error instanceof Error ? error.message : String(error),
            status: 'failed'
          }, null, 2)
        }],
        isError: true
      };
    }
  }

  public async processDeleteObservations(input: unknown): Promise<{ content: Array<{ type: string; text: string }>; isError?: boolean }> {
    try {
      const args = input as { deletions: { entityName: string; observations: string[] }[] };
      await this.deleteObservations(args.deletions);
      return {
        content: [{
          type: "text",
          text: JSON.stringify({ message: "Observations deleted successfully", deletions: args.deletions }, null, 2)
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            error: error instanceof Error ? error.message : String(error),
            status: 'failed'
          }, null, 2)
        }],
        isError: true
      };
    }
  }

  public async processDeleteRelations(input: unknown): Promise<{ content: Array<{ type: string; text: string }>; isError?: boolean }> {
    try {
      const args = input as { relations: Relation[] };
      await this.deleteRelations(args.relations);
      return {
        content: [{
          type: "text",
          text: JSON.stringify({ message: "Relations deleted successfully", deleted: args.relations }, null, 2)
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            error: error instanceof Error ? error.message : String(error),
            status: 'failed'
          }, null, 2)
        }],
        isError: true
      };
    }
  }

  public async processReadGraph(input: unknown): Promise<{ content: Array<{ type: string; text: string }>; isError?: boolean }> {
    try {
      const result = await this.readGraph();
      return {
        content: [{
          type: "text",
          text: JSON.stringify(result, null, 2)
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            error: error instanceof Error ? error.message : String(error),
            status: 'failed'
          }, null, 2)
        }],
        isError: true
      };
    }
  }

  public async processSearchNodes(input: unknown): Promise<{ content: Array<{ type: string; text: string }>; isError?: boolean }> {
    try {
      const args = input as { query: string };
      const result = await this.searchNodes(args.query);
      return {
        content: [{
          type: "text",
          text: JSON.stringify(result, null, 2)
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            error: error instanceof Error ? error.message : String(error),
            status: 'failed'
          }, null, 2)
        }],
        isError: true
      };
    }
  }

  public async processOpenNodes(input: unknown): Promise<{ content: Array<{ type: string; text: string }>; isError?: boolean }> {
    try {
      const args = input as { names: string[] };
      const result = await this.openNodes(args.names);
      return {
        content: [{
          type: "text",
          text: JSON.stringify(result, null, 2)
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            error: error instanceof Error ? error.message : String(error),
            status: 'failed'
          }, null, 2)
        }],
        isError: true
      };
    }
  }
}

// Memory tool definitions with memory_ prefix
export const MEMORY_CREATE_ENTITIES_TOOL: Tool = {
  name: "memory_create_entities",
  description: "Create multiple new entities in the knowledge graph",
  inputSchema: {
    type: "object",
    properties: {
      entities: {
        type: "array",
        items: {
          type: "object",
          properties: {
            name: { type: "string", description: "The name of the entity" },
            entityType: { type: "string", description: "The type of the entity" },
            observations: { 
              type: "array", 
              items: { type: "string" },
              description: "An array of observation contents associated with the entity"
            },
          },
          required: ["name", "entityType", "observations"],
        },
      },
    },
    required: ["entities"],
  },
};

export const MEMORY_CREATE_RELATIONS_TOOL: Tool = {
  name: "memory_create_relations",
  description: "Create multiple new relations between entities in the knowledge graph. Relations should be in active voice",
  inputSchema: {
    type: "object",
    properties: {
      relations: {
        type: "array",
        items: {
          type: "object",
          properties: {
            from: { type: "string", description: "The name of the entity where the relation starts" },
            to: { type: "string", description: "The name of the entity where the relation ends" },
            relationType: { type: "string", description: "The type of the relation" },
          },
          required: ["from", "to", "relationType"],
        },
      },
    },
    required: ["relations"],
  },
};

export const MEMORY_ADD_OBSERVATIONS_TOOL: Tool = {
  name: "memory_add_observations",
  description: "Add new observations to existing entities in the knowledge graph",
  inputSchema: {
    type: "object",
    properties: {
      observations: {
        type: "array",
        items: {
          type: "object",
          properties: {
            entityName: { type: "string", description: "The name of the entity to add the observations to" },
            contents: { 
              type: "array", 
              items: { type: "string" },
              description: "An array of observation contents to add"
            },
          },
          required: ["entityName", "contents"],
        },
      },
    },
    required: ["observations"],
  },
};

export const MEMORY_DELETE_ENTITIES_TOOL: Tool = {
  name: "memory_delete_entities",
  description: "Delete multiple entities and their associated relations from the knowledge graph",
  inputSchema: {
    type: "object",
    properties: {
      entityNames: { 
        type: "array", 
        items: { type: "string" },
        description: "An array of entity names to delete" 
      },
    },
    required: ["entityNames"],
  },
};

export const MEMORY_DELETE_OBSERVATIONS_TOOL: Tool = {
  name: "memory_delete_observations",
  description: "Delete specific observations from entities in the knowledge graph",
  inputSchema: {
    type: "object",
    properties: {
      deletions: {
        type: "array",
        items: {
          type: "object",
          properties: {
            entityName: { type: "string", description: "The name of the entity containing the observations" },
            observations: { 
              type: "array", 
              items: { type: "string" },
              description: "An array of observations to delete"
            },
          },
          required: ["entityName", "observations"],
        },
      },
    },
    required: ["deletions"],
  },
};

export const MEMORY_DELETE_RELATIONS_TOOL: Tool = {
  name: "memory_delete_relations",
  description: "Delete multiple relations from the knowledge graph",
  inputSchema: {
    type: "object",
    properties: {
      relations: { 
        type: "array", 
        items: {
          type: "object",
          properties: {
            from: { type: "string", description: "The name of the entity where the relation starts" },
            to: { type: "string", description: "The name of the entity where the relation ends" },
            relationType: { type: "string", description: "The type of the relation" },
          },
          required: ["from", "to", "relationType"],
        },
        description: "An array of relations to delete" 
      },
    },
    required: ["relations"],
  },
};

export const MEMORY_READ_GRAPH_TOOL: Tool = {
  name: "memory_read_graph",
  description: "Read the entire knowledge graph",
  inputSchema: {
    type: "object",
    properties: {},
  },
};

export const MEMORY_SEARCH_NODES_TOOL: Tool = {
  name: "memory_search_nodes",
  description: "Search for nodes in the knowledge graph based on a query",
  inputSchema: {
    type: "object",
    properties: {
      query: { type: "string", description: "The search query to match against entity names, types, and observation content" },
    },
    required: ["query"],
  },
};

export const MEMORY_OPEN_NODES_TOOL: Tool = {
  name: "memory_open_nodes",
  description: "Open specific nodes in the knowledge graph by their names",
  inputSchema: {
    type: "object",
    properties: {
      names: {
        type: "array",
        items: { type: "string" },
        description: "An array of entity names to retrieve",
      },
    },
    required: ["names"],
  },
};
