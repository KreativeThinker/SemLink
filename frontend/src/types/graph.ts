/**
 * Types for SemLink graph data structures
 */

export interface GraphNode {
  id: string;
  title: string;
  content?: string;
  community?: number;
  centrality?: number;
  topic_id?: number;
  topic_label?: string;
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface GraphEdge {
  source: string | GraphNode;
  target: string | GraphNode;
  weight: number;
  method?: string; // 'threshold', 'knn', 'hybrid', 'hard_link', 'hybrid+hard_link'
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
  method?: string;
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

/**
 * Topic types for semantic grouping
 */

export interface Topic {
  id: number;
  label: string;
  keywords: string[];
  note_ids: string[];
  note_titles: string[];
  size: number;
  central_notes: string[];
}

export interface TopicsData {
  topics: Topic[];
  note_to_topic: Record<string, number>;
  orphan_notes: string[];
}
