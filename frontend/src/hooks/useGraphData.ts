import { useState, useEffect } from 'react';
import type { GraphData, GraphStats } from '../types';

interface UseGraphDataResult {
  data: GraphData | null;
  stats: GraphStats;
  communities: number[];
  loading: boolean;
  error: string | null;
  loadFromFile: (file: File) => Promise<void>;
  loadFromUrl: (url: string) => Promise<void>;
}

export function useGraphData(): UseGraphDataResult {
  const [data, setData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const parseGraphData = (json: any): GraphData => {
    // Handle different JSON formats (D3 format, NetworkX format, etc.)
    let nodes = json.nodes || [];
    let edges = json.edges || json.links || [];

    // Normalize node format
    nodes = nodes.map((n: any) => ({
      id: n.id || n.name,
      title: n.title || n.label || n.id || n.name,
      content: n.content,
      community: n.community ?? n.group,
      centrality: n.centrality ?? n.pagerank,
    }));

    // Normalize edge format
    edges = edges.map((e: any) => ({
      source: e.source,
      target: e.target,
      weight: e.weight ?? e.value ?? 1,
      reason: e.reason,
      shared_terms: e.shared_terms,
    }));

    return { nodes, edges };
  };

  const loadFromFile = async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      const text = await file.text();
      const json = JSON.parse(text);
      const graphData = parseGraphData(json);
      setData(graphData);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load file');
    } finally {
      setLoading(false);
    }
  };

  const loadFromUrl = async (url: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const json = await response.json();
      const graphData = parseGraphData(json);
      setData(graphData);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load URL');
    } finally {
      setLoading(false);
    }
  };

  // Try to load default graph on mount
  useEffect(() => {
    loadFromUrl('/api/graph').catch(() => {
      // API not available, that's fine
    });
  }, []);

  // Calculate stats and communities
  const stats: GraphStats = data
    ? {
        nodeCount: data.nodes.length,
        edgeCount: data.edges.length,
        communities: new Set(data.nodes.map((n) => n.community).filter((c) => c !== undefined)).size,
        avgWeight: data.edges.length > 0
          ? data.edges.reduce((sum, e) => sum + e.weight, 0) / data.edges.length
          : 0,
      }
    : { nodeCount: 0, edgeCount: 0, communities: 0, avgWeight: 0 };

  const communities = data
    ? [...new Set(data.nodes.map((n) => n.community).filter((c): c is number => c !== undefined))].sort((a, b) => a - b)
    : [];

  return {
    data,
    stats,
    communities,
    loading,
    error,
    loadFromFile,
    loadFromUrl,
  };
}
