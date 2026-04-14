import type { GraphNode, GraphEdge } from '../types';

interface SidePanelProps {
  node: GraphNode | null;
  edges: GraphEdge[];
  onClose: () => void;
}

export function SidePanel({ node, edges, onClose }: SidePanelProps) {
  if (!node) return null;

  // Find edges connected to this node
  const connectedEdges = edges.filter(e => {
    const sourceId = typeof e.source === 'string' ? e.source : e.source.id;
    const targetId = typeof e.target === 'string' ? e.target : e.target.id;
    return sourceId === node.id || targetId === node.id;
  });

  return (
    <div className="w-96 h-full bg-slate-800 border-l border-slate-700 overflow-y-auto">
      {/* Header */}
      <div className="sticky top-0 bg-slate-800 border-b border-slate-700 p-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white truncate flex-1 mr-2">
          {node.title}
        </h2>
        <button
          onClick={onClose}
          className="p-1 hover:bg-slate-700 rounded text-slate-400 hover:text-white"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Node info */}
      <div className="p-4 space-y-4">
        {/* Metadata */}
        <div className="flex gap-2 flex-wrap">
          {node.community !== undefined && (
            <span className="px-2 py-1 bg-indigo-600/30 text-indigo-300 text-xs rounded">
              Community {node.community}
            </span>
          )}
          {node.centrality !== undefined && (
            <span className="px-2 py-1 bg-emerald-600/30 text-emerald-300 text-xs rounded">
              Centrality: {node.centrality.toFixed(3)}
            </span>
          )}
        </div>

        {/* Content preview */}
        {node.content && (
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-slate-300">Content</h3>
            <div className="text-sm text-slate-400 bg-slate-900 rounded p-3 max-h-64 overflow-y-auto whitespace-pre-wrap">
              {node.content}
            </div>
          </div>
        )}

        {/* Connections */}
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-slate-300">
            Connections ({connectedEdges.length})
          </h3>
          <div className="space-y-2">
            {connectedEdges
              .sort((a, b) => b.weight - a.weight)
              .map((edge, i) => {
                const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id;
                const targetId = typeof edge.target === 'string' ? edge.target : edge.target.id;
                const otherId = sourceId === node.id ? targetId : sourceId;
                const otherTitle = typeof edge.source === 'string' 
                  ? otherId 
                  : (sourceId === node.id ? (edge.target as GraphNode).title : (edge.source as GraphNode).title);

                return (
                  <div
                    key={i}
                    className="bg-slate-900 rounded p-3 space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white truncate flex-1">
                        {otherTitle}
                      </span>
                      <span className="text-xs text-slate-400 ml-2">
                        {(edge.weight * 100).toFixed(0)}%
                      </span>
                    </div>
                    {edge.reason && (
                      <p className="text-xs text-slate-500">{edge.reason}</p>
                    )}
                    {edge.shared_terms && edge.shared_terms.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {edge.shared_terms.slice(0, 5).map((term, j) => (
                          <span
                            key={j}
                            className="px-1.5 py-0.5 bg-slate-800 text-slate-400 text-xs rounded"
                          >
                            {term}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
          </div>
        </div>
      </div>
    </div>
  );
}
