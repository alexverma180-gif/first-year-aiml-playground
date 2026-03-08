import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables');
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export type Database = {
  public: {
    Tables: {
      user_progress: {
        Row: {
          id: string;
          user_id: string;
          lesson_id: string;
          completed: boolean;
          score: number | null;
          completed_at: string;
          created_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          lesson_id: string;
          completed?: boolean;
          score?: number | null;
          completed_at?: string;
          created_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          lesson_id?: string;
          completed?: boolean;
          score?: number | null;
          completed_at?: string;
          created_at?: string;
        };
      };
      linear_regression_experiments: {
        Row: {
          id: string;
          user_id: string | null;
          learning_rate: number;
          epochs: number;
          num_samples: number;
          num_features: number;
          final_loss: number | null;
          created_at: string;
        };
        Insert: {
          id?: string;
          user_id?: string | null;
          learning_rate: number;
          epochs: number;
          num_samples: number;
          num_features: number;
          final_loss?: number | null;
          created_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string | null;
          learning_rate?: number;
          epochs?: number;
          num_samples?: number;
          num_features?: number;
          final_loss?: number | null;
          created_at?: string;
        };
      };
      iris_predictions: {
        Row: {
          id: string;
          user_id: string | null;
          sepal_length: number;
          sepal_width: number;
          petal_length: number;
          petal_width: number;
          predicted_species: string;
          actual_species: string | null;
          correct: boolean | null;
          created_at: string;
        };
        Insert: {
          id?: string;
          user_id?: string | null;
          sepal_length: number;
          sepal_width: number;
          petal_length: number;
          petal_width: number;
          predicted_species: string;
          actual_species?: string | null;
          correct?: boolean | null;
          created_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string | null;
          sepal_length?: number;
          sepal_width?: number;
          petal_length?: number;
          petal_width?: number;
          predicted_species?: string;
          actual_species?: string | null;
          correct?: boolean | null;
          created_at?: string;
        };
      };
      achievements: {
        Row: {
          id: string;
          user_id: string;
          achievement_type: string;
          title: string;
          description: string;
          earned_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          achievement_type: string;
          title: string;
          description: string;
          earned_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          achievement_type?: string;
          title?: string;
          description?: string;
          earned_at?: string;
        };
      };
    };
  };
};
