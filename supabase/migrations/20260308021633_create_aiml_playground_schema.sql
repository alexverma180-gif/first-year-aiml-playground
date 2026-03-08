/*
  # AIML Playground Database Schema

  1. New Tables
    - `user_progress`
      - `id` (uuid, primary key)
      - `user_id` (uuid, references auth.users)
      - `lesson_id` (text)
      - `completed` (boolean)
      - `score` (integer, nullable)
      - `completed_at` (timestamptz)
      - `created_at` (timestamptz)
    
    - `linear_regression_experiments`
      - `id` (uuid, primary key)
      - `user_id` (uuid, references auth.users, nullable for anonymous)
      - `learning_rate` (numeric)
      - `epochs` (integer)
      - `num_samples` (integer)
      - `num_features` (integer)
      - `final_loss` (numeric)
      - `created_at` (timestamptz)
    
    - `iris_predictions`
      - `id` (uuid, primary key)
      - `user_id` (uuid, references auth.users, nullable for anonymous)
      - `sepal_length` (numeric)
      - `sepal_width` (numeric)
      - `petal_length` (numeric)
      - `petal_width` (numeric)
      - `predicted_species` (text)
      - `actual_species` (text, nullable)
      - `correct` (boolean, nullable)
      - `created_at` (timestamptz)
    
    - `achievements`
      - `id` (uuid, primary key)
      - `user_id` (uuid, references auth.users)
      - `achievement_type` (text)
      - `title` (text)
      - `description` (text)
      - `earned_at` (timestamptz)

  2. Security
    - Enable RLS on all tables
    - Add policies for authenticated users to manage their own data
    - Allow anonymous access for experiments and predictions
*/

-- Create user_progress table
CREATE TABLE IF NOT EXISTS user_progress (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  lesson_id text NOT NULL,
  completed boolean DEFAULT false,
  score integer,
  completed_at timestamptz DEFAULT now(),
  created_at timestamptz DEFAULT now()
);

ALTER TABLE user_progress ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own progress"
  ON user_progress FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own progress"
  ON user_progress FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own progress"
  ON user_progress FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own progress"
  ON user_progress FOR DELETE
  TO authenticated
  USING (auth.uid() = user_id);

-- Create linear_regression_experiments table
CREATE TABLE IF NOT EXISTS linear_regression_experiments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  learning_rate numeric NOT NULL,
  epochs integer NOT NULL,
  num_samples integer NOT NULL,
  num_features integer NOT NULL,
  final_loss numeric,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE linear_regression_experiments ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can insert experiments"
  ON linear_regression_experiments FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

CREATE POLICY "Users can view own experiments"
  ON linear_regression_experiments FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Anonymous can view recent experiments"
  ON linear_regression_experiments FOR SELECT
  TO anon
  USING (created_at > now() - interval '1 hour');

-- Create iris_predictions table
CREATE TABLE IF NOT EXISTS iris_predictions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  sepal_length numeric NOT NULL,
  sepal_width numeric NOT NULL,
  petal_length numeric NOT NULL,
  petal_width numeric NOT NULL,
  predicted_species text NOT NULL,
  actual_species text,
  correct boolean,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE iris_predictions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can insert predictions"
  ON iris_predictions FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

CREATE POLICY "Users can view own predictions"
  ON iris_predictions FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Anonymous can view recent predictions"
  ON iris_predictions FOR SELECT
  TO anon
  USING (created_at > now() - interval '1 hour');

-- Create achievements table
CREATE TABLE IF NOT EXISTS achievements (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  achievement_type text NOT NULL,
  title text NOT NULL,
  description text NOT NULL,
  earned_at timestamptz DEFAULT now()
);

ALTER TABLE achievements ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own achievements"
  ON achievements FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own achievements"
  ON achievements FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_user_progress_user_id ON user_progress(user_id);
CREATE INDEX IF NOT EXISTS idx_user_progress_lesson_id ON user_progress(lesson_id);
CREATE INDEX IF NOT EXISTS idx_experiments_user_id ON linear_regression_experiments(user_id);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON linear_regression_experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON iris_predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON iris_predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_achievements_user_id ON achievements(user_id);