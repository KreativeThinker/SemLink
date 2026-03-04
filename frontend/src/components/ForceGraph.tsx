import { useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import type { GraphData, GraphNode, D3Node, D3Edge, FilterOptions } from '../types';

// Color palette for communities
const COMMUNITY_COLORS = [
  '#6366f1', // indigo
  '#ec4899', // pink
  '#14b8a6', // teal
  '#f59e0b', // amber
  '#8b5cf6', // violet
  '#10b981', // emerald
  '#f43f5e', // rose
  '#06b6d4', // cyan
  '#84cc16', // lime
  '#a855f7', // purple
];

interface ForceGraphProps {
  data: GraphData;
  filters: FilterOptions;
  selectedNode: GraphNode | null;
  onNodeClick: (node: GraphNode) => void;
  onNodeHover: (node: GraphNode | null) => void;
  width: number;
  height: number;
}

export function ForceGraph({
  data,
  filters,
  selectedNode,
  onNodeClick,
  onNodeHover,
  width,
  height,
}: ForceGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const simulationRef = useRef<d3.Simulation<D3Node, D3Edge> | null>(null);

  // Filter data based on current filters
  const filteredData = useCallback(() => {
    let nodes = [...data.nodes];
    let edges = [...data.edges];

    // Filter by search query
    if (filters.searchQuery) {
      const query = filters.searchQuery.toLowerCase();
      const matchingIds = new Set(
        nodes
          .filter(n => 
            n.title.toLowerCase().includes(query) ||
            n.content?.toLowerCase().includes(query)
          )
          .map(n => n.id)
      );
      nodes = nodes.filter(n => matchingIds.has(n.id));
    }

    // Filter by community
    if (filters.community !== null) {
      nodes = nodes.filter(n => n.community === filters.community);
    }

    // Get valid node IDs
    const nodeIds = new Set(nodes.map(n => n.id));

    // Filter edges by weight and valid nodes
    edges = edges.filter(e => {
      const sourceId = typeof e.source === 'string' ? e.source : e.source.id;
      const targetId = typeof e.target === 'string' ? e.target : e.target.id;
      return (
        e.weight >= filters.minWeight &&
        nodeIds.has(sourceId) &&
        nodeIds.has(targetId)
      );
    });

    return { nodes, edges };
  }, [data, filters]);

  useEffect(() => {
    if (!svgRef.current || width === 0 || height === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const { nodes, edges } = filteredData();
    if (nodes.length === 0) return;

    // Create zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Create container for zoom/pan
    const container = svg.append('g');

    // Create arrow marker for directed edges
    svg.append('defs').append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .append('path')
      .attr('d', 'M 0,-5 L 10,0 L 0,5')
      .attr('fill', '#94a3b8');

    // Create simulation
    const simulation = d3.forceSimulation<D3Node>(nodes as D3Node[])
      .force('link', d3.forceLink<D3Node, D3Edge>(edges as D3Edge[])
        .id(d => d.id)
        .distance(100)
        .strength(d => d.weight * 0.5))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    simulationRef.current = simulation;

    // Create edges
    const link = container.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(edges as D3Edge[])
      .join('line')
      .attr('stroke', '#94a3b8')
      .attr('stroke-opacity', d => Math.max(0.2, d.weight))
      .attr('stroke-width', d => Math.max(1, d.weight * 3));

    // Create nodes
    const node = container.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes as D3Node[])
      .join('g')
      .attr('cursor', 'pointer')
      .call(drag(simulation) as any);

    // Add circles to nodes
    node.append('circle')
      .attr('r', d => 8 + (d.centrality || 0) * 20)
      .attr('fill', d => COMMUNITY_COLORS[(d.community || 0) % COMMUNITY_COLORS.length])
      .attr('stroke', d => selectedNode?.id === d.id ? '#ffffff' : 'transparent')
      .attr('stroke-width', 3);

    // Add labels to nodes
    node.append('text')
      .text(d => d.title.length > 20 ? d.title.slice(0, 20) + '...' : d.title)
      .attr('x', 12)
      .attr('y', 4)
      .attr('font-size', '12px')
      .attr('fill', '#e2e8f0')
      .attr('pointer-events', 'none');

    // Add event handlers
    node
      .on('click', (event, d) => {
        event.stopPropagation();
        onNodeClick(d);
      })
      .on('mouseenter', (_, d) => onNodeHover(d))
      .on('mouseleave', () => onNodeHover(null));

    // Click on background to deselect
    svg.on('click', () => onNodeClick(null as any));

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as D3Node).x)
        .attr('y1', d => (d.source as D3Node).y)
        .attr('x2', d => (d.target as D3Node).x)
        .attr('y2', d => (d.target as D3Node).y);

      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [data, filters, filteredData, width, height, selectedNode, onNodeClick, onNodeHover]);

  // Drag behavior
  function drag(simulation: d3.Simulation<D3Node, D3Edge>) {
    function dragstarted(event: d3.D3DragEvent<SVGGElement, D3Node, D3Node>) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: d3.D3DragEvent<SVGGElement, D3Node, D3Node>) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: d3.D3DragEvent<SVGGElement, D3Node, D3Node>) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }

    return d3.drag<SVGGElement, D3Node>()
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended);
  }

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      className="bg-slate-900"
    />
  );
}
