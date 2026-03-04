import { useState, useEffect, useCallback } from 'react';
import type { Topic, TopicsData } from '../types';

interface UseTopicsResult {
  topics: Topic[];
  noteToTopic: Record<string, number>;
  orphanNotes: string[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export function useTopics(): UseTopicsResult {
  const [data, setData] = useState<TopicsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchTopics = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/topics');
      if (!response.ok) {
        if (response.status === 404) {
          // No data available, not an error
          setData({ topics: [], note_to_topic: {}, orphan_notes: [] });
          return;
        }
        throw new Error(`HTTP ${response.status}`);
      }
      const json = await response.json();
      setData(json);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load topics');
    } finally {
      setLoading(false);
    }
  }, []);

  // Load topics on mount
  useEffect(() => {
    fetchTopics();
  }, [fetchTopics]);

  return {
    topics: data?.topics ?? [],
    noteToTopic: data?.note_to_topic ?? {},
    orphanNotes: data?.orphan_notes ?? [],
    loading,
    error,
    refetch: fetchTopics,
  };
}
