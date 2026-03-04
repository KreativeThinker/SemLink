import { useState, useRef, useCallback, useEffect } from 'react';
import { ForceGraph, SidePanel, ControlPanel } from './components';
import { useGraphData } from './hooks';
import type { GraphNode, FilterOptions } from './types';

function App() {
  const { data, stats, communities, loading, error, loadFromFile } = useGraphData();
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [filters, setFilters] = useState<FilterOptions>({
    minWeight: 0,
    community: null,
    searchQuery: '',
  });
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({
          width: rect.width - (selectedNode ? 384 : 0), // Account for side panel
          height: rect.height,
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, [selectedNode]);

  // Handle file drop
  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.name.endsWith('.json')) {
        loadFromFile(file);
      }
    },
    [loadFromFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  // Handle file input
  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        loadFromFile(file);
      }
    },
    [loadFromFile]
  );

  return (
    <div
      ref={containerRef}
      className="w-full h-screen flex bg-slate-900"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
    >
      {/* Main graph area */}
      <div className="flex-1 relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80 z-50">
            <div className="text-white text-lg">Loading...</div>
          </div>
        )}

        {error && (
          <div className="absolute top-4 right-4 bg-red-900/90 text-red-200 px-4 py-2 rounded z-50">
            {error}
          </div>
        )}

        {!data ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div className="text-center space-y-4">
              <div className="text-6xl">📊</div>
              <h2 className="text-2xl font-bold text-white">SemLink Graph Viewer</h2>
              <p className="text-slate-400 max-w-md">
                Drop a graph JSON file here or click below to load one.
                <br />
                Use <code className="bg-slate-800 px-1 rounded">semlink visualize -f d3</code> to generate.
              </p>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-6 py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-medium transition-colors"
              >
                Load Graph File
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".json"
                onChange={handleFileChange}
                className="hidden"
              />
            </div>
          </div>
        ) : (
          <>
            <ForceGraph
              data={data}
              filters={filters}
              selectedNode={selectedNode}
              onNodeClick={setSelectedNode}
              onNodeHover={setHoveredNode}
              width={dimensions.width}
              height={dimensions.height}
            />

            <ControlPanel
              filters={filters}
              stats={stats}
              communities={communities}
              onFiltersChange={setFilters}
            />

            {/* Hover tooltip */}
            {hoveredNode && !selectedNode && (
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-slate-800/95 backdrop-blur px-4 py-2 rounded-lg shadow-xl border border-slate-700">
                <span className="text-white font-medium">{hoveredNode.title}</span>
                {hoveredNode.community !== undefined && (
                  <span className="ml-2 text-xs text-slate-400">
                    Community {hoveredNode.community}
                  </span>
                )}
              </div>
            )}
          </>
        )}
      </div>

      {/* Side panel */}
      {selectedNode && data && (
        <SidePanel
          node={selectedNode}
          edges={data.edges}
          onClose={() => setSelectedNode(null)}
        />
      )}
    </div>
  );
}

export default App;
