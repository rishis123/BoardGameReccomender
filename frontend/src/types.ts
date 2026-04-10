export interface Ingredient {
  id: number;
  name: string;
  category: string;
}

export interface Substitution extends Ingredient {
  similarity: number;
  shared_molecules: Molecule[];
}

export interface Molecule {
  id: number;
  pubchem_id: string;
  common_name: string;
  flavor_profile: string;
}

export interface IngredientProfile extends Ingredient {
  scientific_name: string;
  molecule_count: number;
  molecules: Molecule[];
}

export interface PMIEdge {
  source: number;
  target: number;
  pmi: number;
  shared_molecules: string[];
}

export interface NetworkData {
  nodes: Ingredient[];
  edges: PMIEdge[];
}

export interface SensoryPoint {
  id: number;
  name: string;
  category: string;
  x: number;
  y: number;
  z: number;
}

export interface DimensionInfo {
  index: number;
  label: string;
  explained_variance: number;
  top_molecules: { molecule_id: number; name: string; loading: number }[];
}

export interface SensoryMapData {
  points: SensoryPoint[];
  dimensions: DimensionInfo[];
}

export interface MetricsData {
  avg_feedback: number;
  total_feedback: number;
  precision_at_k?: number;
  [key: string]: number | undefined;
}

export interface ChatMessage {
  text: string;
  isUser: boolean;
  citations?: string[];
  id?: string;
}

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
    seed?: {
      id: string;
      name: string;
    } | null;
  };
  recommendations: RecommendationResult[];
  latent_dimensions: LatentDimension[];
}
