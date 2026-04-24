export interface GameSuggestion {
  id: string;
  name: string;
  year_published: number | null;
  users_rated: number;
}

export interface WhyTag {
  index: number;
  label: string;
  activation: number;
}

export interface TermLoading {
  term: string;
  loading: number;
}

export interface LatentDimension {
  index: number;
  label: string;
  explained_variance: number;
  terms: TermLoading[];
}

export interface RecommendationResult {
  id: string;
  name: string;
  snippet: string;
  thumbnail: string | null;
  year_published: number | null;
  average_rating: number | null;
  users_rated: number;
  category: string;
  mechanic: string;
  score_svd: number;
  score_tfidf: number;
  rank_svd: number;
  rank_tfidf: number;
  why_tags: WhyTag[];
}

export interface RecommendationResponse {
  query: {
    method?: 'svd' | 'tfidf';
    text?: string | null;
    seed?: { id: string; name: string } | null;
  };
  recommendations: RecommendationResult[];
  latent_dimensions: LatentDimension[];
}

export interface QueryDimension {
  index: number;
  label: string;
  activation: number;
  terms: TermLoading[];
}

export interface RagResponse {
  original_label: string;
  original_dims: QueryDimension[];
  rewritten_query: string;
  rewritten_dims: QueryDimension[];
  original_results: RecommendationResult[];
  rag_results: RecommendationResult[];
  llm_summary: string | null;
  error: string | null;
}
