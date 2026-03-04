/**
 * Types for SemLink graph data structures
 */

export interface GraphNode {
  id: string;
  title: string;
  content?: string;
  community?: number;
  centrality?: number;
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface GraphEdge {
  source: string | GraphNode;
  target: string | GraphNode;
  weight: number;
  reason?: string;
  shared_terms?: string[];
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface D3Node extends GraphNode {
  x: number;
  y: number;
  vx?: number;
  vy?: number;
}

export interface D3Edge {
  source: D3Node;
  target: D3Node;
  weight: number;
  reason?: string;
  shared_terms?: string[];
}

export interface FilterOptions {
  minWeight: number;
  community: number | null;
  searchQuery: string;
}

export interface GraphStats {
  nodeCount: number;
  edgeCount: number;
  communities: number;
  avgWeight: number;
}
