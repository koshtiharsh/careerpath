import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input
import joblib
import json
import os
import datetime
import time

def generate_expanded_career_data():
    """Generate a more diverse career dataset with 100 unique careers"""
    import numpy as np
    import pandas as pd
    
    # List of unique career titles with their required skills
    careers = [
        # Technology
        {'title': 'Software Engineer', 'required_skills': 'python,java,algorithms,data structures,problem solving', 'field': 'Technology', 'education_level': 'Bachelor'},
        {'title': 'Data Scientist', 'required_skills': 'python,statistics,machine learning,data analysis,visualization', 'field': 'Technology', 'education_level': 'Master'},
        {'title': 'Frontend Developer', 'required_skills': 'javascript,html,css,react,ui design', 'field': 'Technology', 'education_level': 'Bachelor'},
        {'title': 'Backend Developer', 'required_skills': 'java,spring,node.js,databases,api design', 'field': 'Technology', 'education_level': 'Bachelor'},
        {'title': 'DevOps Engineer', 'required_skills': 'docker,kubernetes,jenkins,aws,automation', 'field': 'Technology', 'education_level': 'Bachelor'},
        {'title': 'Cloud Architect', 'required_skills': 'aws,azure,cloud infrastructure,networking,security', 'field': 'Technology', 'education_level': 'Master'},
        {'title': 'Cybersecurity Analyst', 'required_skills': 'network security,penetration testing,incident response,security tools', 'field': 'Technology', 'education_level': 'Bachelor'},
        {'title': 'Machine Learning Engineer', 'required_skills': 'python,tensorflow,neural networks,feature engineering,optimization', 'field': 'Technology', 'education_level': 'Master'},
        {'title': 'Mobile App Developer', 'required_skills': 'swift,kotlin,react native,mobile ui design,api integration', 'field': 'Technology', 'education_level': 'Bachelor'},
        {'title': 'Database Administrator', 'required_skills': 'sql,database optimization,data modeling,backup recovery,oracle', 'field': 'Technology', 'education_level': 'Bachelor'},
        {'title': 'AI Research Scientist', 'required_skills': 'machine learning,research,mathematics,neural networks,algorithms', 'field': 'Technology', 'education_level': 'PhD'},
        {'title': 'Blockchain Developer', 'required_skills': 'blockchain,solidity,smart contracts,cryptography,web3', 'field': 'Technology', 'education_level': 'Bachelor'},
        {'title': 'Game Developer', 'required_skills': 'unity,c#,game design,3d modeling,physics', 'field': 'Technology', 'education_level': 'Bachelor'},
        {'title': 'Quantum Computing Researcher', 'required_skills': 'quantum mechanics,linear algebra,algorithms,research,python', 'field': 'Technology', 'education_level': 'PhD'},
        {'title': 'AR/VR Developer', 'required_skills': 'unity,3d modeling,c#,ui/ux design,spatial computing', 'field': 'Technology', 'education_level': 'Bachelor'},
        
        # Business
        {'title': 'Product Manager', 'required_skills': 'product development,roadmap planning,communication,leadership,analysis', 'field': 'Business', 'education_level': 'Bachelor'},
        {'title': 'Business Analyst', 'required_skills': 'sql,data analysis,requirements gathering,process modeling,excel', 'field': 'Business', 'education_level': 'Bachelor'},
        {'title': 'Project Manager', 'required_skills': 'project planning,budgeting,leadership,risk management,communication', 'field': 'Business', 'education_level': 'Bachelor'},
        {'title': 'Management Consultant', 'required_skills': 'business strategy,problem solving,presentation,analytics,industry knowledge', 'field': 'Business', 'education_level': 'Master'},
        {'title': 'Financial Analyst', 'required_skills': 'financial modeling,excel,accounting,valuation,statistics', 'field': 'Business', 'education_level': 'Bachelor'},
        {'title': 'Investment Banker', 'required_skills': 'financial modeling,valuation,market analysis,deal structuring,excel', 'field': 'Business', 'education_level': 'Bachelor'},
        {'title': 'Operations Manager', 'required_skills': 'process optimization,leadership,resource management,analytics,logistics', 'field': 'Business', 'education_level': 'Bachelor'},
        {'title': 'Supply Chain Manager', 'required_skills': 'logistics,inventory management,procurement,analytics,optimization', 'field': 'Business', 'education_level': 'Bachelor'},
        {'title': 'HR Manager', 'required_skills': 'recruitment,employee relations,hr policies,benefits administration,leadership', 'field': 'Business', 'education_level': 'Bachelor'},
        {'title': 'Strategy Director', 'required_skills': 'strategic planning,market analysis,business development,leadership,analytics', 'field': 'Business', 'education_level': 'Master'},
        
        # Design
        {'title': 'UX Designer', 'required_skills': 'wireframing,prototyping,user research,visual design,interaction design', 'field': 'Design', 'education_level': 'Bachelor'},
        {'title': 'UI Designer', 'required_skills': 'visual design,typography,color theory,interaction design,figma', 'field': 'Design', 'education_level': 'Bachelor'},
        {'title': 'Graphic Designer', 'required_skills': 'adobe creative suite,typography,composition,branding,illustration', 'field': 'Design', 'education_level': 'Bachelor'},
        {'title': 'Product Designer', 'required_skills': 'user research,wireframing,prototyping,visual design,interaction design', 'field': 'Design', 'education_level': 'Bachelor'},
        {'title': 'Motion Designer', 'required_skills': 'after effects,animation principles,storyboarding,visual design,3d', 'field': 'Design', 'education_level': 'Bachelor'},
        {'title': 'Industrial Designer', 'required_skills': 'cad,3d modeling,product design,prototyping,materials knowledge', 'field': 'Design', 'education_level': 'Bachelor'},
        {'title': 'Design Director', 'required_skills': 'design strategy,team leadership,creative direction,client management,design thinking', 'field': 'Design', 'education_level': 'Bachelor'},
        
        # Marketing
        {'title': 'Marketing Specialist', 'required_skills': 'social media,content creation,analytics,campaign management,SEO', 'field': 'Marketing', 'education_level': 'Bachelor'},
        {'title': 'Digital Marketing Manager', 'required_skills': 'sem,seo,social media,analytics,content strategy', 'field': 'Marketing', 'education_level': 'Bachelor'},
        {'title': 'Brand Manager', 'required_skills': 'brand strategy,market research,product positioning,campaign management,analytics', 'field': 'Marketing', 'education_level': 'Bachelor'},
        {'title': 'SEO Specialist', 'required_skills': 'keyword research,link building,content optimization,analytics,technical seo', 'field': 'Marketing', 'education_level': 'Bachelor'},
        {'title': 'Social Media Manager', 'required_skills': 'content creation,community management,analytics,paid social,strategy', 'field': 'Marketing', 'education_level': 'Bachelor'},
        {'title': 'Growth Marketer', 'required_skills': 'acquisition strategy,funnel optimization,ab testing,analytics,user behavior', 'field': 'Marketing', 'education_level': 'Bachelor'},
        {'title': 'Content Strategist', 'required_skills': 'content planning,seo,audience research,writing,editorial calendar', 'field': 'Marketing', 'education_level': 'Bachelor'},
        
        # Healthcare
        {'title': 'Registered Nurse', 'required_skills': 'patient care,medical knowledge,critical thinking,communication,documentation', 'field': 'Healthcare', 'education_level': 'Bachelor'},
        {'title': 'Physician Assistant', 'required_skills': 'clinical care,diagnosis,medical knowledge,patient management,communication', 'field': 'Healthcare', 'education_level': 'Master'},
        {'title': 'Physical Therapist', 'required_skills': 'rehabilitation techniques,anatomy,patient assessment,treatment planning,documentation', 'field': 'Healthcare', 'education_level': 'Master'},
        {'title': 'Medical Laboratory Technician', 'required_skills': 'specimen collection,lab equipment,test procedures,data analysis,accuracy', 'field': 'Healthcare', 'education_level': 'Associate'},
        {'title': 'Healthcare Administrator', 'required_skills': 'healthcare regulations,management,budget,policy development,leadership', 'field': 'Healthcare', 'education_level': 'Bachelor'},
        {'title': 'Nurse Practitioner', 'required_skills': 'patient assessment,diagnosis,treatment planning,prescribing,healthcare knowledge', 'field': 'Healthcare', 'education_level': 'Master'},
        {'title': 'Health Informatics Specialist', 'required_skills': 'electronic health records,data analysis,healthcare systems,it,compliance', 'field': 'Healthcare', 'education_level': 'Bachelor'},
        
        # Education
        {'title': 'Elementary Teacher', 'required_skills': 'curriculum development,classroom management,communication,assessment,pedagogy', 'field': 'Education', 'education_level': 'Bachelor'},
        {'title': 'Special Education Teacher', 'required_skills': 'individualized education plans,behavior management,assessment,adaptations,communication', 'field': 'Education', 'education_level': 'Bachelor'},
        {'title': 'Curriculum Designer', 'required_skills': 'instructional design,assessment,learning theories,content development,education standards', 'field': 'Education', 'education_level': 'Master'},
        {'title': 'School Counselor', 'required_skills': 'counseling techniques,student support,academic guidance,assessment,communication', 'field': 'Education', 'education_level': 'Master'},
        {'title': 'University Professor', 'required_skills': 'research,teaching,subject expertise,curriculum development,academic writing', 'field': 'Education', 'education_level': 'PhD'},
        {'title': 'Education Technology Specialist', 'required_skills': 'digital learning tools,technology integration,training,instructional design,assessment', 'field': 'Education', 'education_level': 'Bachelor'},
        
        # Finance
        {'title': 'Accountant', 'required_skills': 'financial reporting,gaap,tax preparation,auditing,accounting software', 'field': 'Finance', 'education_level': 'Bachelor'},
        {'title': 'Financial Planner', 'required_skills': 'investment analysis,retirement planning,tax planning,client communication,financial regulations', 'field': 'Finance', 'education_level': 'Bachelor'},
        {'title': 'Risk Analyst', 'required_skills': 'risk assessment,data analysis,financial modeling,reporting,compliance', 'field': 'Finance', 'education_level': 'Bachelor'},
        {'title': 'Investment Analyst', 'required_skills': 'financial modeling,market research,valuation,excel,investment strategies', 'field': 'Finance', 'education_level': 'Bachelor'},
        {'title': 'Actuary', 'required_skills': 'statistical analysis,risk assessment,mathematics,insurance,financial modeling', 'field': 'Finance', 'education_level': 'Bachelor'},
        {'title': 'Quantitative Analyst', 'required_skills': 'mathematics,programming,statistics,financial markets,algorithms', 'field': 'Finance', 'education_level': 'Master'},
        
        # Legal
        {'title': 'Attorney', 'required_skills': 'legal research,writing,analysis,client counseling,negotiation', 'field': 'Legal', 'education_level': 'Master'},
        {'title': 'Paralegal', 'required_skills': 'legal research,document preparation,case management,organization,communication', 'field': 'Legal', 'education_level': 'Associate'},
        {'title': 'Compliance Officer', 'required_skills': 'regulatory knowledge,risk assessment,policy development,auditing,reporting', 'field': 'Legal', 'education_level': 'Bachelor'},
        {'title': 'Contract Manager', 'required_skills': 'contract review,negotiation,legal knowledge,documentation,relationship management', 'field': 'Legal', 'education_level': 'Bachelor'},
        
        # Science & Research
        {'title': 'Research Scientist', 'required_skills': 'research methods,data analysis,scientific writing,experimentation,subject expertise', 'field': 'Science', 'education_level': 'PhD'},
        {'title': 'Biomedical Engineer', 'required_skills': 'medical device design,biology,engineering principles,testing,research', 'field': 'Science', 'education_level': 'Master'},
        {'title': 'Environmental Scientist', 'required_skills': 'environmental analysis,field research,data collection,report writing,regulations', 'field': 'Science', 'education_level': 'Bachelor'},
        {'title': 'Chemist', 'required_skills': 'laboratory techniques,analytical methods,research,data analysis,safety protocols', 'field': 'Science', 'education_level': 'Bachelor'},
        {'title': 'Biotechnologist', 'required_skills': 'molecular biology,lab techniques,research,data analysis,biotechnology applications', 'field': 'Science', 'education_level': 'Master'},
        
        # Engineering
        {'title': 'Civil Engineer', 'required_skills': 'structural analysis,cad,project management,construction knowledge,regulations', 'field': 'Engineering', 'education_level': 'Bachelor'},
        {'title': 'Mechanical Engineer', 'required_skills': 'mechanical design,cad,thermodynamics,materials science,problem solving', 'field': 'Engineering', 'education_level': 'Bachelor'},
        {'title': 'Electrical Engineer', 'required_skills': 'circuit design,power systems,electronics,control systems,cad', 'field': 'Engineering', 'education_level': 'Bachelor'},
        {'title': 'Aerospace Engineer', 'required_skills': 'aerodynamics,mechanical design,materials science,cad,thermodynamics', 'field': 'Engineering', 'education_level': 'Bachelor'},
        {'title': 'Chemical Engineer', 'required_skills': 'process engineering,thermodynamics,reaction kinetics,transport phenomena,plant design', 'field': 'Engineering', 'education_level': 'Bachelor'},
        
        # Creative
        {'title': 'Content Writer', 'required_skills': 'writing,editing,research,seo,content strategy', 'field': 'Creative', 'education_level': 'Bachelor'},
        {'title': 'Video Editor', 'required_skills': 'adobe premiere,after effects,storytelling,audio editing,color grading', 'field': 'Creative', 'education_level': 'Bachelor'},
        
        {'title': 'Photographer', 'required_skills': 'photography techniques,adobe photoshop,lighting,composition,visual storytelling', 'field': 'Creative', 'education_level': 'Bachelor'},
        {'title': 'Animator', 'required_skills': 'animation principles,after effects,character design,storyboarding,3d modeling', 'field': 'Creative', 'education_level': 'Bachelor'},
        {'title': 'Copywriter', 'required_skills': 'writing,advertising,brand voice,creativity,editing', 'field': 'Creative', 'education_level': 'Bachelor'},
        {'title': 'Art Director', 'required_skills': 'visual design,creative direction,team leadership,conceptualization,branding', 'field': 'Creative', 'education_level': 'Bachelor'},
        {'title': 'Journalist', 'required_skills': 'reporting,writing,research,interviewing,editing', 'field': 'Creative', 'education_level': 'Bachelor'},
    ]
    
    # Convert the list of careers to a DataFrame
    df = pd.DataFrame(careers)
    
    # Generate salary ranges based on education level
    salary_ranges = {
        'Associate': (40000, 65000),
        'Bachelor': (55000, 95000),
        'Master': (70000, 120000),
        'PhD': (90000, 160000)
    }
    
    # Add salary data
    df['salary_low'] = df['education_level'].apply(lambda x: salary_ranges[x][0])
    df['salary_high'] = df['education_level'].apply(lambda x: salary_ranges[x][1])
    df['avg_salary'] = (df['salary_low'] + df['salary_high']) / 2
    
    # Add years of experience requirement (randomly assigned)
    np.random.seed(42)  # For reproducibility
    df['min_experience'] = np.random.choice([0, 1, 2, 3, 5], size=len(df))
    
    # Add job satisfaction rating (1-10 scale)
    df['job_satisfaction'] = np.random.uniform(6.0, 9.0, size=len(df)).round(1)
    
    # Add growth outlook percentage (expected job growth in next decade)
    df['growth_outlook'] = np.random.uniform(-5.0, 25.0, size=len(df)).round(1)
    
    return df
class AdaptiveCareerAI:
    """
    AdaptiveCareer AI: A self-improving career recommendation model that learns 
    from user feedback to continuously enhance prediction accuracy.
    """
    def __init__(self, model_path=None):
        # Core components
        self.skills_model = None
        self.market_model = None
        self.preference_model = None
        self.feedback_model = None
        self.career_data = None
        self.job_market_data = None
        self.feedback_data = []
        
        # Model version tracking
        self.version = "1.0.0"
        self.last_trained = datetime.datetime.now().strftime("%Y-%m-%d")
        self.feedback_count = 0
        self.accuracy_score = 0.0
        
        # Load existing model or initialize a new one
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            # Load datasets
            self._load_datasets()
            # Initialize models
            self._initialize_models()
    
    def _load_datasets(self):
        """Load initial career and job market datasets"""
        # In a real system, this would connect to real data sources
        
        # Generate expanded career dataset using the new function
        career_list = generate_expanded_career_data()
        
        # Convert the list to a DataFrame and add additional fields
        career_df = pd.DataFrame(career_list)
        
        # Add career_id and additional metrics
        career_df['career_id'] = range(1, len(career_df) + 1)
        
        # Add the numeric fields that weren't in your original function
        career_df['average_salary'] = np.random.randint(60000, 180000, len(career_df))
        career_df['growth_rate'] = np.random.uniform(0.02, 0.25, len(career_df))
        career_df['work_life_balance'] = np.random.uniform(2.5, 4.5, len(career_df))
        career_df['remote_opportunities'] = np.random.uniform(0.1, 0.9, len(career_df))
        
        # Generate career paths
        career_df['career_path'] = career_df['title'].apply(
            lambda title: f"Junior {title} -> {title} -> Senior {title} -> Lead {title} -> Director/Head of {title.split()[-1]}"
        )
        
        self.career_data = career_df
        
        # Job market data remains the same but adjust for the new number of careers
        num_careers = len(career_df)
        num_regions = 5
        job_market_entries = num_careers * num_regions
        
        self.job_market_data = pd.DataFrame({
            'region': ['Northeast', 'West', 'South', 'Midwest', 'Remote'] * num_careers,
            'career_id': np.repeat(range(1, num_careers + 1), num_regions),
            'demand_score': np.random.uniform(0.2, 0.9, job_market_entries),
            'competition_level': np.random.uniform(0.3, 0.8, job_market_entries),
            'salary_trend': np.random.uniform(-0.05, 0.15, job_market_entries),
            'job_openings': np.random.randint(500, 10000, job_market_entries)
        })
        
        # Initialize feedback data storage
        self.feedback_data = []
    
    def _initialize_models(self):
        """Initialize all AI models from scratch"""
        # Skills model - using TF-IDF and cosine similarity
        self.skills_vectorizer = TfidfVectorizer(stop_words='english')
        self.skills_matrix = self.skills_vectorizer.fit_transform(self.career_data['required_skills'])
        
        # Market analysis model - Neural network
        market_inputs = Input(shape=(5,))
        x = Dense(16, activation='relu')(market_inputs)
        x = Dense(8, activation='relu')(x)
        market_output = Dense(1, activation='sigmoid')(x)
        self.market_model = Model(inputs=market_inputs, outputs=market_output)
        self.market_model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Preference matching model - KMeans clustering
        self.preference_clusters = KMeans(n_clusters=5)
        preference_features = self.career_data[['average_salary', 'growth_rate', 'work_life_balance', 'remote_opportunities']]
        self.preference_clusters.fit(preference_features)
        self.career_data['preference_cluster'] = self.preference_clusters.labels_
        
        # Feedback learning model - Initialize GradientBoosting regressor
        self.feedback_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
        # This will be trained once we have enough feedback data
    
    def _save_model(self, path="./adaptive_career_model"):
        """Save the trained model and all components"""
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Save vectorizer
        joblib.dump(self.skills_vectorizer, f"{path}/skills_vectorizer.pkl")
        
        # Save neural network
        self.market_model.save(f"{path}/market_model.h5")
        
        # Save clustering model
        joblib.dump(self.preference_clusters, f"{path}/preference_clusters.pkl")
        
        # Save feedback model if trained
        if hasattr(self.feedback_model, 'feature_importances_'):
            joblib.dump(self.feedback_model, f"{path}/feedback_model.pkl")
        
        # Save datasets
        self.career_data.to_csv(f"{path}/career_data.csv", index=False)
        self.job_market_data.to_csv(f"{path}/job_market_data.csv", index=False)
        
        # Save feedback data
        feedback_df = pd.DataFrame(self.feedback_data)
        if not feedback_df.empty:
            feedback_df.to_csv(f"{path}/feedback_data.csv", index=False)
        
        # Save metadata
        metadata = {
            "version": self.version,
            "last_trained": self.last_trained,
            "feedback_count": self.feedback_count,
            "accuracy_score": self.accuracy_score
        }
        
        with open(f"{path}/metadata.json", "w") as f:
            json.dump(metadata, f)
            
        print(f"Model saved to {path}")
    
    def _load_model(self, path):
        """Load a previously saved model"""
        try:
            # Load vectorizer
            self.skills_vectorizer = joblib.load(f"{path}/skills_vectorizer.pkl")
            
            # Load neural network
            self.market_model = load_model(f"{path}/market_model.h5")
            
            # Load clustering model
            self.preference_clusters = joblib.load(f"{path}/preference_clusters.pkl")
            
            # Load feedback model if exists
            if os.path.exists(f"{path}/feedback_model.pkl"):
                self.feedback_model = joblib.load(f"{path}/feedback_model.pkl")
            else:
                self.feedback_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
            
            # Load datasets
            self.career_data = pd.read_csv(f"{path}/career_data.csv")
            self.job_market_data = pd.read_csv(f"{path}/job_market_data.csv")
            
            # Load feedback data if exists
            if os.path.exists(f"{path}/feedback_data.csv"):
                self.feedback_data = pd.read_csv(f"{path}/feedback_data.csv").to_dict('records')
            else:
                self.feedback_data = []
            
            # Load metadata
            with open(f"{path}/metadata.json", "r") as f:
                metadata = json.load(f)
                self.version = metadata.get("version", "1.0.0")
                self.last_trained = metadata.get("last_trained", datetime.datetime.now().strftime("%Y-%m-%d"))
                self.feedback_count = metadata.get("feedback_count", 0)
                self.accuracy_score = metadata.get("accuracy_score", 0.0)
            
            # Recompute skills matrix
            self.skills_matrix = self.skills_vectorizer.transform(self.career_data['required_skills'])
            
            print(f"Model loaded from {path} (version {self.version})")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self._load_datasets()
            self._initialize_models()
            return False
    
    def analyze_user_skills(self, user_skills):
        """Analyze user skills and match to potential careers"""
        # Transform user skills
        user_skills_vector = self.skills_vectorizer.transform([user_skills])
        
        # Calculate similarity with all careers
        similarities = cosine_similarity(user_skills_vector, self.skills_matrix).flatten()
        
        # Get top matching careers
        top_indices = similarities.argsort()[-10:][::-1]
        skills_matches = self.career_data.iloc[top_indices].copy()
        skills_matches['skill_match_score'] = similarities[top_indices]
        
        return skills_matches
    
    def analyze_market_fit(self, career_ids, user_region):
        """Analyze market fit for potential careers"""
        # Filter job market data for the specific region and careers
        regional_market = self.job_market_data[
            (self.job_market_data['region'] == user_region) & 
            (self.job_market_data['career_id'].isin(career_ids))
        ]
        
        if regional_market.empty:
            regional_market = self.job_market_data[self.job_market_data['career_id'].isin(career_ids)]
        
        # Calculate market score using proper pandas operations to avoid SettingWithCopyWarning
        # Create a copy to avoid the warning
        regional_market = regional_market.copy()
        regional_market['market_score'] = (
            (regional_market['demand_score'] * 0.4) + 
            (1 - regional_market['competition_level'] * 0.3) + 
            (regional_market['salary_trend'] * 20 * 0.3)
        )
        return regional_market[['career_id', 'demand_score', 'competition_level', 'salary_trend', 'job_openings', 'market_score']]
    
    def analyze_preferences(self, user_preferences):
        """Match user preferences to careers"""
        # Normalize preferences
        total = sum(user_preferences.values())
        normalized_prefs = {k: v/total for k, v in user_preferences.items()}
        
        # Create a copy to avoid modifying the original DataFrame
        preference_data = self.career_data[['career_id']].copy()
        
        # Calculate preference scores for all careers
        preference_data['preference_score'] = (
            self.career_data['average_salary'] / 150000 * normalized_prefs['salary_importance'] +
            self.career_data['growth_rate'] * normalized_prefs['growth_importance'] +
            self.career_data['work_life_balance'] / 5 * normalized_prefs['balance_importance'] +
            self.career_data['remote_opportunities'] * normalized_prefs['remote_importance']
        )
        
        return preference_data

    def predict_career_recommendations(self, user_data):
        """Generate recommendations based on user data, including feedback-based adjustments"""
        # Extract user information
        user_skills = user_data['skills']
        user_region = user_data['region']
        user_preferences = user_data['preferences']
        user_education = user_data['education_level']
        
        # Get skill matches
        skill_matches = self.analyze_user_skills(user_skills)
        
        # Filter by education requirements
        education_levels = {
            'High School': 0,
            'Associate': 1,
            'Bachelor': 2,
            'Master': 3,
            'PhD': 4
        }
        
        user_edu_level = education_levels.get(user_education, 0)
        edu_levels_map = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
        
        # Convert education to numeric for comparison
        skill_matches['edu_numeric'] = skill_matches['education_level'].map(edu_levels_map)
        
        # Keep careers where user meets education requirements or is one level below
        skill_matches = skill_matches[skill_matches['edu_numeric'] <= user_edu_level + 1]
        
        if skill_matches.empty:
            return {
                'success': False,
                'message': 'No suitable career matches found based on your skills and education.',
                'recommendations': [],
                'model_version': self.version,
                'confidence': 0.0
            }
        
        # Get career IDs for market analysis
        career_ids = skill_matches['career_id'].tolist()
        
        # Analyze market fit
        market_analysis = self.analyze_market_fit(career_ids, user_region)
        
        # Analyze preference match
        preference_match = self.analyze_preferences(user_preferences)
        
        # Make sure preference_match contains all career_ids from skill_matches
        valid_career_ids = set(career_ids).intersection(set(preference_match['career_id']))
        if len(valid_career_ids) < len(career_ids):
            print(f"Warning: Some careers ({len(career_ids) - len(valid_career_ids)}) not found in preference analysis")
        
        # Combine all scores
        merged_data = skill_matches.merge(market_analysis, on='career_id')
        merged_data = merged_data.merge(preference_match, on='career_id')
        
        # Base scores (before feedback model adjustment)
        merged_data['base_score'] = (
            merged_data['skill_match_score'] * 0.4 +
            merged_data['market_score'] * 0.3 +
            merged_data['preference_score'] * 0.3
        )
        
        # If feedback model has been trained, apply it to adjust scores
        if hasattr(self.feedback_model, 'feature_importances_') and len(self.feedback_data) >= 10:
            # Prepare features for feedback model
            features = []
            for _, row in merged_data.iterrows():
                # Extract features that would affect satisfaction
                features.append([
                    row['skill_match_score'],
                    row['market_score'],
                    row['preference_score'],
                    row['average_salary'] / 150000,
                    row['growth_rate'],
                    row['work_life_balance'] / 5,
                    row['remote_opportunities'],
                    education_levels.get(row['education_level'], 2) / 4
                ])
            
            # Predict satisfaction adjustment
            satisfaction_adjustments = self.feedback_model.predict(features)
            
            # Apply adjustments to final score (weighted by confidence in feedback model)
            confidence = min(self.feedback_count / 100, 0.5)  # Cap at 0.5 to keep base model relevant
            merged_data['final_score'] = merged_data['base_score'] * (1 - confidence) + satisfaction_adjustments * confidence
        else:
            # Without feedback model, use base score
            merged_data['final_score'] = merged_data['base_score']
        
        # Sort by final score
        merged_data = merged_data.sort_values('final_score', ascending=False)
        
        # Top recommendations
        top_recommendations = merged_data.head(5)
        
        # Generate skill gaps analysis for top career
        top_career = top_recommendations.iloc[0]
        required_skills = set(top_career['required_skills'].split(','))
        user_skills_set = set(user_skills.split(','))
        skill_gaps = list(required_skills - user_skills_set)
        
        # Calculate confidence score based on data quality and feedback quantity
        confidence_score = 0.5  # Base confidence
        confidence_score += min(self.feedback_count / 200, 0.3)  # More feedback improves confidence
        confidence_score += min(len(skill_matches) / 20, 0.1)  # More skill matches improve confidence
        confidence_score += min(top_career['skill_match_score'] * 0.2, 0.1)  # Higher top match improves confidence
        
        # Prepare recommendation response
        recommendations = []
        for _, career in top_recommendations.iterrows():
            recommendations.append({
                'title': career['title'],
                'match_score': round(career['final_score'] * 100, 1),
                'skill_match': round(career['skill_match_score'] * 100, 1),
                'market_outlook': round(career['market_score'] * 100, 1),
                'preference_match': round(career['preference_score'] * 100, 1),
                'average_salary': career['average_salary'],
                'growth_rate': f"{career['growth_rate'] * 100:.1f}%",
                'job_openings': int(career['job_openings']),
                'career_path': career['career_path'],
                'education_required': career['education_level'],
                'recommendation_id': f"rec_{career['career_id']}_{datetime.datetime.now().strftime('%Y%m%d')}"
            })
        
        return {
            'success': True,
            'recommendations': recommendations,
            'skill_gaps': skill_gaps,
            'development_paths': self._generate_development_paths(skill_gaps, top_career['title']),
            'model_version': self.version,
            'confidence': round(confidence_score, 2),
            'feedback_requested': True
        }
    
    def _generate_development_paths(self, skill_gaps, career_title):
        """Generate development paths to address skill gaps"""
        development_paths = []
        
        for skill in skill_gaps:
            development_paths.append({
                'skill': skill,
                'resources': [
                    {
                        'type': 'Online Course',
                        'title': f'Master {skill} for {career_title} Professionals',
                        'provider': 'Coursera',
                        'duration': '8 weeks',
                        'cost': '$49'
                    },
                    {
                        'type': 'Certification',
                        'title': f'Professional {skill} Certification',
                        'provider': 'Industry Association',
                        'duration': '3 months',
                        'cost': '$299'
                    }
                ]
            })
        
        return development_paths
    
    def add_user_feedback(self, feedback_data):
        """
        Add user feedback to improve model accuracy
        
        Parameters:
        feedback_data (dict): Feedback from the user with the following structure:
            {
                'recommendation_id': str,  # ID of the recommendation
                'user_id': str,           # Unique user identifier
                'career_title': str,      # Career title
                'satisfaction_score': float, # User satisfaction (0-10)
                'user_data': dict,        # Original user data used for prediction
                'comments': str,          # Optional user comments
                'followed_recommendation': bool  # Whether user followed the recommendation
            }
        """
        # Add timestamp
        feedback_data['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Store feedback
        self.feedback_data.append(feedback_data)
        self.feedback_count += 1
        
        # Check if we need to retrain the feedback model
        if self.feedback_count % 10 == 0 and self.feedback_count >= 10:
            self._retrain_feedback_model()
            
            # Update version with minor increment
            version_parts = self.version.split('.')
            version_parts[-1] = str(int(version_parts[-1]) + 1)
            self.version = '.'.join(version_parts)
            self.last_trained = datetime.datetime.now().strftime("%Y-%m-%d")
            
            # Save updated model
            self._save_model()
            
        return {
            'success': True,
            'feedback_count': self.feedback_count,
            'message': 'Feedback recorded successfully. Thank you for helping improve our model!'
        }
    
    def _retrain_feedback_model(self):
        """Retrain the feedback model based on accumulated user feedback"""
        if len(self.feedback_data) < 10:
            return False
        
        print("Retraining feedback model with new data...")
        
        # Prepare training data from feedback
        X_train = []
        y_train = []
        
        for feedback in self.feedback_data:
            user_data = feedback['user_data']
            career_title = feedback['career_title']
            satisfaction = feedback['satisfaction_score']
            
            # Get career details
            career_info = self.career_data[self.career_data['title'] == career_title]
            if career_info.empty:
                continue
                
            career_info = career_info.iloc[0]
            
            # Get skill match score
            user_skills = user_data['skills']
            user_skills_vector = self.skills_vectorizer.transform([user_skills])
            skill_match = cosine_similarity(user_skills_vector, 
                                          self.skills_vectorizer.transform([career_info['required_skills']]))[0][0]
            
            # Get market score
            region = user_data['region']
            market_info = self.job_market_data[
                (self.job_market_data['region'] == region) & 
                (self.job_market_data['career_id'] == career_info['career_id'])
            ]
            
            if market_info.empty:
                market_info = self.job_market_data[self.job_market_data['career_id'] == career_info['career_id']].iloc[0]
            else:
                market_info = market_info.iloc[0]
                
            market_score = (
                (market_info['demand_score'] * 0.4) + 
                (1 - market_info['competition_level'] * 0.3) + 
                (market_info['salary_trend'] * 20 * 0.3)
            )
            
            # Get preference score
            preferences = user_data['preferences']
            total = sum(preferences.values())
            normalized_prefs = {k: v/total for k, v in preferences.items()}
            
            preference_score = (
                career_info['average_salary'] / 150000 * normalized_prefs['salary_importance'] +
                career_info['growth_rate'] * normalized_prefs['growth_importance'] +
                career_info['work_life_balance'] / 5 * normalized_prefs['balance_importance'] +
                career_info['remote_opportunities'] * normalized_prefs['remote_importance']
            )
            
            # Build feature vector
            education_levels = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
            feature_vector = [
                skill_match,
                market_score,
                preference_score,
                career_info['average_salary'] / 150000,
                career_info['growth_rate'],
                career_info['work_life_balance'] / 5,
                career_info['remote_opportunities'],
                education_levels.get(career_info['education_level'], 2) / 4
            ]
            
            X_train.append(feature_vector)
            y_train.append(satisfaction / 10.0)  # Normalize to 0-1
        
        # Split into training and validation sets
        if len(X_train) > 20:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
            # Train model
            self.feedback_model.fit(X_train, y_train)
            
            # Evaluate model
            self.accuracy_score = self.feedback_model.score(X_val, y_val)
            print(f"Feedback model retrained. Validation score: {self.accuracy_score:.4f}")
        else:
            # For small datasets, use all data
            self.feedback_model.fit(X_train, y_train)
            self.accuracy_score = self.feedback_model.score(X_train, y_train)
            print(f"Feedback model retrained. Training score: {self.accuracy_score:.4f}")
        
        return True
    
    def get_model_stats(self):
        """Get model statistics and version information"""
        return {
            'version': self.version,
            'last_trained': self.last_trained,
            'feedback_count': self.feedback_count,
            'accuracy_score': round(self.accuracy_score, 4),
            'model_type': 'AdaptiveCareerAI',
            'core_models': {
                'skills_model': 'TF-IDF with Cosine Similarity',
                'market_model': 'Neural Network',
                'preference_model': 'K-Means Clustering',
                'feedback_model': 'Gradient Boosting Regressor'
            },
            'data_stats': {
                'careers': len(self.career_data),
                'market_regions': self.job_market_data['region'].nunique(),
                'skill_vocabulary_size': len(self.skills_vectorizer.get_feature_names_out()) if hasattr(self.skills_vectorizer, 'get_feature_names_out') else 0
            }
        }


# Interactive testing interface for AdaptiveCareerAI
class CareerAITester:
    """Interactive testing interface to test individual components of AdaptiveCareerAI"""
    
    def __init__(self, model_path=None):
        self.model = AdaptiveCareerAI(model_path)
        self.test_user_data = {
            'skills': 'python,data analysis,statistics,visualization,machine learning',
            'education_level': 'Bachelor',
            'region': 'Northeast',
            'preferences': {
                'salary_importance': 0.3,
                'growth_importance': 0.3, 
                'balance_importance': 0.2,
                'remote_importance': 0.2
            }
        }
        self.last_recommendations = None
    
    def display_menu(self):
        """Display interactive menu options"""
        print("\n" + "="*50)
        print("ADAPTIVE CAREER AI - INTERACTIVE TESTING MENU")
        print("="*50)
        print("1. Test Skills Analysis")
        print("2. Test Market Analysis")
        print("3. Test Preference Analysis")
        print("4. Get Career Recommendations")
        print("5. Submit User Feedback")
        print("6. Retrain Model with Feedback")
        print("7. Update User Profile")
        print("8. Save Model")
        print("9. Load Model")
        print("10. View Model Statistics")
        print("11. Generate Simulated Feedback")
        print("12. Exit")
        print("="*50)
        
        choice = input("Enter your choice (1-12): ")
        return choice
    
    def run(self):
        """Run the interactive testing interface"""
        while True:
            choice = self.display_menu()
            
            if choice == '1':
                self.test_skills_analysis()
            elif choice == '2':
                self.test_market_analysis()
            elif choice == '3':
                self.test_preference_analysis()
            elif choice == '4':
                self.test_recommendations()
            elif choice == '5':
                self.test_submit_feedback()
            elif choice == '6':
                self.test_retrain_model()
            elif choice == '7':
                self.update_user_profile()
            elif choice == '8':
                self.save_model()
            elif choice == '9':
                self.load_model()
            elif choice == '10':
                self.view_model_stats()
            elif choice == '11':
                self.generate_simulated_feedback()
            elif choice == '12':
                print("Exiting. Thank you for using AdaptiveCareerAI!")
                break
            else:
                print("Invalid choice. Please select 1-12.")
            
            input("\nPress Enter to continue...")
    
    def test_skills_analysis(self):
        """Test the skills analysis component"""
        print("\n==== SKILLS ANALYSIS TEST ====")
        
        # Use default skills or get new ones
        use_default = input(f"Use default skills '{self.test_user_data['skills']}'? (y/n): ").lower()
        
        if use_default != 'y':
            skills = input("Enter skills (comma separated): ")
            self.test_user_data['skills'] = skills
        
        # Run skills analysis
        print("\nAnalyzing skills...")
        matches = self.model.analyze_user_skills(self.test_user_data['skills'])
        
        # Display results
        print("\nTop Career Matches:")
        print(f"{'Career Title':<20} | {'Skill Match %':<15} | {'Required Skills':<30}")
        print("-" * 70)
        
        for _, match in matches[:5].iterrows():
            print(f"{match['title']:<20} | {match['skill_match_score']*100:.1f}%{' ':10} | {match['required_skills'][:30]}...")
    
    def test_market_analysis(self):
        """Test the market analysis component"""
        print("\n==== MARKET ANALYSIS TEST ====")
        
        # Select region
        regions = ['Northeast', 'West', 'South', 'Midwest', 'Remote']
        
        print("\nAvailable regions:")
        for i, region in enumerate(regions, 1):
            print(f"{i}. {region}")
            
        region_choice = input(f"Select region (1-5) [default: {self.test_user_data['region']}]: ")
        
        if region_choice and region_choice.isdigit() and 1 <= int(region_choice) <= 5:
            region = regions[int(region_choice) - 1]
            self.test_user_data['region'] = region
        else:
            region = self.test_user_data['region']
        
        # Get some career IDs
        # Get some career IDs
        skills_matches = self.model.analyze_user_skills(self.test_user_data['skills'])
        career_ids = skills_matches['career_id'].tolist()[:5]  # Get top 5 matching careers
        
        # Run market analysis
        print(f"\nAnalyzing market for region: {region}...")
        market_data = self.model.analyze_market_fit(career_ids, region)
        
        # Display results
        careers = self.model.career_data[self.model.career_data['career_id'].isin(career_ids)]
        
        print("\nMarket Analysis Results:")
        print(f"{'Career Title':<20} | {'Demand':<10} | {'Competition':<15} | {'Salary Trend':<15} | {'Openings':<10} | {'Score':<10}")
        print("-" * 90)
        
        for _, market in market_data.iterrows():
            career = careers[careers['career_id'] == market['career_id']].iloc[0]
            print(f"{career['title']:<20} | {market['demand_score']*100:>6.1f}%{' ':3} | {market['competition_level']*100:>9.1f}%{' ':5} | {market['salary_trend']*100:>9.1f}%{' ':5} | {int(market['job_openings']):>8} | {market['market_score']*100:>6.1f}%")
    
    def test_preference_analysis(self):
        """Test the preference analysis component"""
        print("\n==== PREFERENCE ANALYSIS TEST ====")
        
        # Show current preferences
        print("\nCurrent preferences:")
        for pref, value in self.test_user_data['preferences'].items():
            print(f"{pref}: {value:.1f}")
        
        # Update preferences?
        update_prefs = input("\nUpdate preferences? (y/n): ").lower()
        
        if update_prefs == 'y':
            print("\nEnter importance values (0-10):")
            salary_imp = float(input("Salary importance: ")) / 10
            growth_imp = float(input("Growth importance: ")) / 10
            balance_imp = float(input("Work-life balance importance: ")) / 10
            remote_imp = float(input("Remote work importance: ")) / 10
            
            # Normalize
            total = salary_imp + growth_imp + balance_imp + remote_imp
            self.test_user_data['preferences'] = {
                'salary_importance': salary_imp / total,
                'growth_importance': growth_imp / total,
                'balance_importance': balance_imp / total,
                'remote_importance': remote_imp / total
            }
        
        # Run preference analysis
        print("\nAnalyzing preferences...")
        
        # Get some careers to analyze
        skills_matches = self.model.analyze_user_skills(self.test_user_data['skills'])
        top_careers = self.model.career_data[self.model.career_data['career_id'].isin(skills_matches['career_id'].tolist()[:10])]
        
        # Get preference scores
        preference_data = self.model.analyze_preferences(self.test_user_data['preferences'])
        
        # Filter to our selected careers
        preference_data = preference_data[preference_data['career_id'].isin(top_careers['career_id'])]
        
        # Merge with career data
        result = top_careers.merge(preference_data, on='career_id')
        result = result.sort_values('preference_score', ascending=False)
        
        # Display results
        print("\nPreference Match Results:")
        print(f"{'Career Title':<20} | {'Salary':<10} | {'Growth':<10} | {'Balance':<10} | {'Remote':<10} | {'Score':<10}")
        print("-" * 80)
        
        for _, row in result[:5].iterrows():
            print(f"{row['title']:<20} | ${row['average_salary']:>8,} | {row['growth_rate']*100:>6.1f}%{' ':3} | {row['work_life_balance']:>6.1f}/5{' ':2} | {row['remote_opportunities']*100:>6.1f}%{' ':3} | {row['preference_score']*100:>6.1f}%")
    
    def test_recommendations(self):
        """Test the complete recommendation engine"""
        print("\n==== CAREER RECOMMENDATIONS TEST ====")
        
        # Show current user profile
        print("\nCurrent User Profile:")
        print(f"Skills: {self.test_user_data['skills']}")
        print(f"Education: {self.test_user_data['education_level']}")
        print(f"Region: {self.test_user_data['region']}")
        print("Preferences:")
        for pref, value in self.test_user_data['preferences'].items():
            print(f"  - {pref}: {value:.1f}")
        
        # Get recommendations
        print("\nGenerating recommendations...")
        results = self.model.predict_career_recommendations(self.test_user_data)
        self.last_recommendations = results  # Store for feedback
        
        if not results['success']:
            print(f"\nError: {results['message']}")
            return
        
        # Display recommendations
        print(f"\nTop Career Recommendations (Model v{results['model_version']}, Confidence: {results['confidence']*100:.1f}%):")
        print("-" * 80)
        
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec['title']}")
            print(f"   Overall Match: {rec['match_score']}%")
            print(f"   Skill Match: {rec['skill_match']}%")
            print(f"   Market Outlook: {rec['market_outlook']}%")
            print(f"   Preference Match: {rec['preference_match']}%")
            print(f"   Salary: ${rec['average_salary']:,}")
            print(f"   Growth Rate: {rec['growth_rate']}")
            print(f"   Job Openings: {rec['job_openings']:,}")
            print(f"   Education Required: {rec['education_required']}")
            print(f"   Career Path: {rec['career_path']}")
            print(f"   Recommendation ID: {rec['recommendation_id']}")
            print("-" * 80)
        
        # Show skill gaps
        if results['skill_gaps']:
            print("\nSkill Gaps for Top Recommendation:")
            for skill in results['skill_gaps']:
                print(f"- {skill}")
            
            print("\nDevelopment Paths:")
            for path in results['development_paths'][:3]:  # Show first 3 paths
                print(f"Skill: {path['skill']}")
                for resource in path['resources']:
                    print(f"  - {resource['type']}: {resource['title']} ({resource['provider']}, {resource['duration']}, {resource['cost']})")
    
    def test_submit_feedback(self):
        """Test submitting user feedback"""
        print("\n==== SUBMIT USER FEEDBACK TEST ====")
        
        if not self.last_recommendations or not self.last_recommendations.get('success', False):
            print("Error: No recommendations have been generated yet. Please run 'Get Career Recommendations' first.")
            return
        
        # List recommendations for selection
        print("\nSelect recommendation to provide feedback for:")
        recommendations = self.last_recommendations['recommendations']
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']} (Match: {rec['match_score']}%)")
        
        # Get selection
        choice = input("\nEnter choice (1-5): ")
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(recommendations):
            print("Invalid choice.")
            return
        
        selected_rec = recommendations[int(choice)-1]
        
        # Get feedback data
        print(f"\nProviding feedback for: {selected_rec['title']}")
        satisfaction = float(input("Satisfaction score (0-10): "))
        followed = input("Would you follow this recommendation? (y/n): ").lower() == 'y'
        comments = input("Any comments? (optional): ")
        
        # Create feedback object
        feedback = {
            'recommendation_id': selected_rec['recommendation_id'],
            'user_id': f"test_user_{int(time.time())}",
            'career_title': selected_rec['title'],
            'satisfaction_score': satisfaction,
            'user_data': self.test_user_data,
            'comments': comments,
            'followed_recommendation': followed
        }
        
        # Submit feedback
        result = self.model.add_user_feedback(feedback)
        
        # Show result
        print(f"\nFeedback submission result: {result['message']}")
        print(f"Total feedback count: {result['feedback_count']}")
    
    def test_retrain_model(self):
        """Force retrain the feedback model"""
        print("\n==== RETRAIN MODEL TEST ====")
        
        if len(self.model.feedback_data) < 5:
            print(f"Insufficient feedback data ({len(self.model.feedback_data)} entries). Need at least 5.")
            return
        
        print(f"Retraining model with {len(self.model.feedback_data)} feedback entries...")
        
        # Force retrain
        self.model._retrain_feedback_model()
        
        # Show stats
        print(f"\nModel retrained (version {self.model.version})")
        print(f"Accuracy score: {self.model.accuracy_score:.4f}")
        print(f"Last trained: {self.model.last_trained}")
    
    def update_user_profile(self):
        """Update the test user profile"""
        print("\n==== UPDATE USER PROFILE ====")
        
        print("\nCurrent user profile:")
        for key, value in self.test_user_data.items():
            if key != 'preferences':
                print(f"{key}: {value}")
            else:
                print(f"{key}:")
                for pref, val in value.items():
                    print(f"  - {pref}: {val:.1f}")
        
        # What to update?
        print("\nWhat would you like to update?")
        print("1. Skills")
        print("2. Education level")
        print("3. Region")
        print("4. Preferences")
        print("5. All of the above")
        
        choice = input("\nEnter choice (1-5): ")
        
        if choice == '1' or choice == '5':
            self.test_user_data['skills'] = input("Enter skills (comma separated): ")
            
        if choice == '2' or choice == '5':
            print("\nEducation levels:")
            print("1. High School")
            print("2. Associate")
            print("3. Bachelor")
            print("4. Master")
            print("5. PhD")
            edu_choice = input("Select education level (1-5): ")
            edu_map = {
                '1': 'High School',
                '2': 'Associate',
                '3': 'Bachelor',
                '4': 'Master',
                '5': 'PhD'
            }
            if edu_choice in edu_map:
                self.test_user_data['education_level'] = edu_map[edu_choice]
            
        if choice == '3' or choice == '5':
            print("\nRegions:")
            regions = ['Northeast', 'West', 'South', 'Midwest', 'Remote']
            for i, region in enumerate(regions, 1):
                print(f"{i}. {region}")
            region_choice = input("Select region (1-5): ")
            if region_choice.isdigit() and 1 <= int(region_choice) <= 5:
                self.test_user_data['region'] = regions[int(region_choice) - 1]
            
        if choice == '4' or choice == '5':
            print("\nEnter importance values (0-10):")
            salary_imp = float(input("Salary importance: ")) / 10
            growth_imp = float(input("Growth importance: ")) / 10
            balance_imp = float(input("Work-life balance importance: ")) / 10
            remote_imp = float(input("Remote work importance: ")) / 10
            
            # Normalize
            total = salary_imp + growth_imp + balance_imp + remote_imp
            self.test_user_data['preferences'] = {
                'salary_importance': salary_imp / total,
                'growth_importance': growth_imp / total,
                'balance_importance': balance_imp / total,
                'remote_importance': remote_imp / total
            }
        
        print("\nUser profile updated!")
    
    def save_model(self):
        """Save the current model"""
        print("\n==== SAVE MODEL ====")
        
        path = input("Enter save path (default: './adaptive_career_model'): ")
        if not path:
            path = './adaptive_career_model'
            
        self.model._save_model(path)
    
    def load_model(self):
        """Load a saved model"""
        print("\n==== LOAD MODEL ====")
        
        path = input("Enter model path (default: './adaptive_career_model'): ")
        if not path:
            path = './adaptive_career_model'
            
        success = self.model._load_model(path)
        
        if success:
            print(f"Model loaded successfully (version {self.model.version})")
        else:
            print("Error loading model. New model initialized.")
    
    def view_model_stats(self):
        """View model statistics"""
        print("\n==== MODEL STATISTICS ====")
        
        stats = self.model.get_model_stats()
        
        print(f"Model Type: {stats['model_type']}")
        print(f"Version: {stats['version']}")
        print(f"Last Trained: {stats['last_trained']}")
        print(f"Feedback Count: {stats['feedback_count']}")
        print(f"Accuracy Score: {stats['accuracy_score']:.4f}")
        
        print("\nCore Models:")
        for model_name, model_type in stats['core_models'].items():
            print(f"- {model_name}: {model_type}")
        
        print("\nData Statistics:")
        for stat_name, stat_value in stats['data_stats'].items():
            print(f"- {stat_name}: {stat_value}")
    
    def generate_simulated_feedback(self):
        """Generate simulated feedback to train the model faster"""
        print("\n==== GENERATE SIMULATED FEEDBACK ====")
        
        count = input("How many feedback entries to generate? (1-50): ")
        if not count.isdigit() or int(count) < 1 or int(count) > 50:
            print("Invalid count. Using 10.")
            count = 10
        else:
            count = int(count)
        
        print(f"\nGenerating {count} simulated feedback entries...")
        
        # Get all career titles
        career_titles = self.model.career_data['title'].unique()
        
        # Education levels
        education_levels = ['High School', 'Associate', 'Bachelor', 'Master', 'PhD']
        
        # Regions
        regions = ['Northeast', 'West', 'South', 'Midwest', 'Remote']
        
        # Generate feedback
        # Generate feedback
        for i in range(count):
            # Random career title
            career_title = np.random.choice(career_titles)
            
            # Get career info
            career_info = self.model.career_data[self.model.career_data['title'] == career_title].iloc[0]
            
            # Generate random user data
            user_data = {
                'skills': career_info['required_skills'],  # Start with perfect match
                'education_level': np.random.choice(education_levels),
                'region': np.random.choice(regions),
                'preferences': {
                    'salary_importance': np.random.uniform(0.1, 0.4),
                    'growth_importance': np.random.uniform(0.1, 0.4),
                    'balance_importance': np.random.uniform(0.1, 0.4),
                    'remote_importance': np.random.uniform(0.1, 0.4)
                }
            }
            
            # Normalize preferences
            total = sum(user_data['preferences'].values())
            user_data['preferences'] = {k: v/total for k, v in user_data['preferences'].items()}
            
            # Sometimes reduce skill match by removing skills
            if np.random.random() < 0.7:  # 70% chance
                skills = user_data['skills'].split(',')
                # Remove 10-30% of skills
                remove_count = max(1, int(len(skills) * np.random.uniform(0.1, 0.3)))
                for _ in range(remove_count):
                    if len(skills) > 2:  # Keep at least 2 skills
                        skills.pop(np.random.randint(0, len(skills)))
                user_data['skills'] = ','.join(skills)
            
            # Calculate expected satisfaction
            # Higher for better skill match, education match, and preference alignment
            base_satisfaction = np.random.uniform(5.0, 9.0)  # Base satisfaction 5-9
            
            # Adjust based on skill match (simulated)
            user_skills_set = set(user_data['skills'].split(','))
            career_skills_set = set(career_info['required_skills'].split(','))
            skill_match_pct = len(user_skills_set.intersection(career_skills_set)) / len(career_skills_set)
            satisfaction_adj = (skill_match_pct - 0.5) * 2  # -1 to +1 adjustment
            
            # Adjust for education
            edu_levels = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
            user_edu = edu_levels.get(user_data['education_level'], 2)
            career_edu = edu_levels.get(career_info['education_level'], 2)
            edu_adj = -0.5 if user_edu < career_edu else 0.2 if user_edu > career_edu else 0
            
            # Final satisfaction
            satisfaction = base_satisfaction + satisfaction_adj + edu_adj
            satisfaction = max(0, min(10, satisfaction))  # Clamp to 0-10
            
            # Create feedback entry
            feedback = {
                'recommendation_id': f"sim_rec_{career_info['career_id']}_{int(time.time())}_{i}",
                'user_id': f"sim_user_{int(time.time())}_{i}",
                'career_title': career_title,
                'satisfaction_score': satisfaction,
                'user_data': user_data,
                'comments': "Simulated feedback entry",
                'followed_recommendation': np.random.random() < 0.7  # 70% chance of following
            }
            
            # Add feedback
            self.model.add_user_feedback(feedback)
            
            # Progress update
            if (i + 1) % 5 == 0 or i == count - 1:
                print(f"Generated {i + 1}/{count} feedback entries...")
        
        print(f"\nCompleted generation of {count} simulated feedback entries.")
        print(f"Total feedback count: {self.model.feedback_count}")
        
        # Ask to retrain
        if input("\nRetrain model with new feedback? (y/n): ").lower() == 'y':
            self.test_retrain_model()


# Main execution
if __name__ == "__main__":
    print("AdaptiveCareerAI - Interactive Testing Tool")
    print("------------------------------------------")
    
    load_path = input("Enter path to load existing model (or press Enter for new model): ")
    
    tester = CareerAITester(load_path if load_path else None)
    tester.run()