-- Advanced Calculator Database Schema
-- PostgreSQL with advanced features for AI-powered calculator

-- User Management Tables
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    subscription_type VARCHAR(50) DEFAULT 'free'
);

-- Student Verification System
CREATE TABLE student_verifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    institution_name VARCHAR(255),
    student_id VARCHAR(100),
    education_level VARCHAR(50),
    field_of_study VARCHAR(100),
    graduation_year INTEGER,
    verification_status VARCHAR(20) DEFAULT 'pending',
    verification_method VARCHAR(50),
    verification_documents JSONB,
    verified_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User Profiles and Preferences
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE UNIQUE,
    avatar_url VARCHAR(500),
    bio TEXT,
    preferred_units VARCHAR(20) DEFAULT 'metric',
    default_precision INTEGER DEFAULT 10,
    preferred_notation VARCHAR(20) DEFAULT 'standard',
    theme VARCHAR(20) DEFAULT 'light',
    language VARCHAR(10) DEFAULT 'en',
    timezone VARCHAR(50) DEFAULT 'UTC',
    notification_preferences JSONB DEFAULT '{"email": true, "push": true, "achievements": true}',
    privacy_settings JSONB DEFAULT '{"profile_visible": true, "history_visible": false}'
);

-- Calculation History and Sessions
CREATE TABLE calculation_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_name VARCHAR(255),
    session_type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_shared BOOLEAN DEFAULT FALSE,
    share_token VARCHAR(100) UNIQUE,
    tags TEXT[],
    metadata JSONB
);

CREATE TABLE calculations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES calculation_sessions(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    calculation_type VARCHAR(100) NOT NULL,
    input_data JSONB NOT NULL,
    input_method VARCHAR(50),
    processed_input JSONB,
    calculation_steps JSONB,
    result JSONB NOT NULL,
    execution_time_ms INTEGER,
    ai_confidence_score DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_bookmarked BOOLEAN DEFAULT FALSE,
    tags TEXT[],
    notes TEXT,
    difficulty_level INTEGER CHECK (difficulty_level >= 1 AND difficulty_level <= 10)
);

-- Problem Recognition and AI Features
CREATE TABLE problem_recognitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id UUID REFERENCES calculations(id) ON DELETE CASCADE,
    recognition_type VARCHAR(50),
    original_input BYTEA,
    recognition_confidence DECIMAL(3,2),
    processing_time_ms INTEGER,
    ai_model_used VARCHAR(100),
    recognized_text TEXT,
    recognized_equations JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Collaborative Features
CREATE TABLE collaboration_rooms (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    creator_id UUID REFERENCES users(id) ON DELETE CASCADE,
    room_code VARCHAR(20) UNIQUE NOT NULL,
    max_participants INTEGER DEFAULT 10,
    is_active BOOLEAN DEFAULT TRUE,
    room_type VARCHAR(50) DEFAULT 'study_group',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE room_participants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    room_id UUID REFERENCES collaboration_rooms(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(20) DEFAULT 'participant',
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(room_id, user_id)
);

-- Achievement and Gamification System
CREATE TABLE achievements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    difficulty VARCHAR(20),
    icon_url VARCHAR(500),
    points INTEGER DEFAULT 0,
    requirements JSONB,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE user_achievements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    achievement_id UUID REFERENCES achievements(id) ON DELETE CASCADE,
    earned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    progress JSONB,
    UNIQUE(user_id, achievement_id)
);

-- Learning Analytics and Predictions
CREATE TABLE learning_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    subject VARCHAR(100),
    skill_level DECIMAL(4,2) CHECK (skill_level >= 0 AND skill_level <= 100),
    problems_solved INTEGER DEFAULT 0,
    average_time_per_problem INTEGER,
    accuracy_rate DECIMAL(3,2),
    learning_velocity DECIMAL(5,2),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    prediction_data JSONB,
    UNIQUE(user_id, subject)
);

-- Formula and Concept Library
CREATE TABLE formulas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    formula_latex TEXT NOT NULL,
    category VARCHAR(100),
    subcategory VARCHAR(100),
    description TEXT,
    variables JSONB,
    constants JSONB,
    difficulty_level INTEGER CHECK (difficulty_level >= 1 AND difficulty_level <= 10),
    usage_count INTEGER DEFAULT 0,
    tags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Custom User Formulas
CREATE TABLE user_formulas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    formula_id UUID REFERENCES formulas(id) ON DELETE CASCADE,
    custom_name VARCHAR(255),
    custom_variables JSONB,
    is_favorite BOOLEAN DEFAULT FALSE,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, formula_id)
);

-- API Usage and Rate Limiting
CREATE TABLE api_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    endpoint VARCHAR(255),
    method VARCHAR(10),
    calculations_count INTEGER DEFAULT 0,
    ai_requests_count INTEGER DEFAULT 0,
    reset_date DATE DEFAULT CURRENT_DATE,
    monthly_limit INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, endpoint, reset_date)
);

-- Error Logging and System Monitoring
CREATE TABLE calculation_errors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    calculation_id UUID REFERENCES calculations(id) ON DELETE SET NULL,
    error_type VARCHAR(100),
    error_message TEXT,
    stack_trace TEXT,
    input_data JSONB,
    system_info JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Feedback and Rating System
CREATE TABLE calculation_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id UUID REFERENCES calculations(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    is_result_correct BOOLEAN,
    suggestions TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(calculation_id, user_id)
);

-- Indexes for Performance Optimization
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_subscription_type ON users(subscription_type);
CREATE INDEX idx_calculations_user_id ON calculations(user_id);
CREATE INDEX idx_calculations_type ON calculations(calculation_type);
CREATE INDEX idx_calculations_created_at ON calculations(created_at);
CREATE INDEX idx_calculation_sessions_user_id ON calculation_sessions(user_id);
CREATE INDEX idx_learning_analytics_user_id ON learning_analytics(user_id);
CREATE INDEX idx_formulas_category ON formulas(category);
CREATE INDEX idx_user_achievements_user_id ON user_achievements(user_id);

-- Full-text search indexes
CREATE INDEX idx_formulas_search ON formulas USING gin(to_tsvector('english', name || ' ' || description));
CREATE INDEX idx_calculations_search ON calculations USING gin(to_tsvector('english', input_data::text));

-- Triggers for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sessions_updated_at BEFORE UPDATE ON calculation_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Views for common queries
CREATE VIEW user_statistics AS
SELECT 
    u.id,
    u.username,
    u.subscription_type,
    COUNT(c.id) as total_calculations,
    COUNT(DISTINCT c.calculation_type) as unique_calculation_types,
    AVG(c.ai_confidence_score) as avg_confidence_score,
    COUNT(ua.achievement_id) as total_achievements,
    u.created_at as user_since
FROM users u
LEFT JOIN calculations c ON u.id = c.user_id
LEFT JOIN user_achievements ua ON u.id = ua.user_id
GROUP BY u.id, u.username, u.subscription_type, u.created_at;

CREATE VIEW popular_formulas AS
SELECT 
    f.*,
    COUNT(uf.user_id) as user_count,
    AVG(cf.rating) as avg_rating
FROM formulas f
LEFT JOIN user_formulas uf ON f.id = uf.formula_id
LEFT JOIN calculations c ON f.id::text = c.input_data->>'formula_id'
LEFT JOIN calculation_feedback cf ON c.id = cf.calculation_id
GROUP BY f.id
ORDER BY user_count DESC, avg_rating DESC;

-- Sample Data Insertion
INSERT INTO achievements (name, description, category, difficulty, points, requirements) VALUES
('First Calculation', 'Complete your first calculation', 'beginner', 'bronze', 10, '{"calculations_count": 1}'),
('Problem Solver', 'Solve 100 problems', 'problem_solver', 'silver', 100, '{"calculations_count": 100}'),
('Math Wizard', 'Solve 1000 math problems', 'problem_solver', 'gold', 500, '{"calculations_count": 1000, "category": "mathematics"}'),
('Physics Expert', 'Master physics calculations', 'subject_expert', 'gold', 750, '{"physics_accuracy": 0.95, "physics_problems": 200}'),
('Collaborator', 'Help others in study rooms', 'social', 'silver', 200, '{"collaboration_sessions": 10}');

-- Insert sample formulas
INSERT INTO formulas (name, formula_latex, category, subcategory, description, variables, difficulty_level) VALUES
('Quadratic Formula', 'x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}', 'mathematics', 'algebra', 'Solves quadratic equations ax² + bx + c = 0', '{"a": "coefficient of x²", "b": "coefficient of x", "c": "constant term"}', 3),
('Newton''s Second Law', 'F = ma', 'physics', 'mechanics', 'Force equals mass times acceleration', '{"F": "force (N)", "m": "mass (kg)", "a": "acceleration (m/s²)"}', 2),
('Einstein Mass-Energy', 'E = mc^2', 'physics', 'relativity', 'Mass-energy equivalence', '{"E": "energy (J)", "m": "mass (kg)", "c": "speed of light (m/s)"}', 4),
('Compound Interest', 'A = P(1 + \\frac{r}{n})^{nt}', 'finance', 'interest', 'Calculates compound interest', '{"A": "final amount", "P": "principal", "r": "annual rate", "n": "compounds per year", "t": "time in years"}', 3);
