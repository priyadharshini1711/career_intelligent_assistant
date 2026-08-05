"""A curated skill vocabulary.

Why a dictionary at all, when an LLM could extract skills?

Because this vocabulary drives the *fit score*, and a score that changes
between two runs of the same documents is worthless. Dictionary matching is
deterministic, instant, free, and inspectable -- when the UI says "you're
missing Kubernetes", there is a specific line in a specific chunk behind it.
The LLM still does the reasoning in chat; it just does not get to decide the
number.

The obvious cost is coverage: a skill outside this list is invisible to the
matcher. That is mitigated two ways -- `extract_unlisted_requirements` in
skills.py falls back to noun-phrase mining for requirement bullets that match
nothing here, and the semantic-alignment component of the fit score works on
embeddings and so does not depend on the vocabulary at all.

Format: (canonical name, category, aliases, ambiguous)

`ambiguous` marks names that are real words or too short to match safely
("Go", "R", "C"). Those are only matched inside a skills-style context -- see
`_ambiguous_context` in skills.py -- because otherwise every mention of "go to
market" scores as Golang.
"""

from typing import Dict, List, NamedTuple, Tuple


class SkillDef(NamedTuple):
    canonical: str
    category: str
    aliases: Tuple[str, ...]
    ambiguous: bool = False


_RAW: List[SkillDef] = [
    # -- programming languages -------------------------------------------
    SkillDef("Python", "language", ("python3", "py")),
    SkillDef("JavaScript", "language", ("js", "ecmascript")),
    SkillDef("TypeScript", "language", ("ts",)),
    SkillDef("Java", "language", ()),
    SkillDef("Go", "language", ("golang",), ambiguous=True),
    SkillDef("C++", "language", ("cpp", "c plus plus")),
    SkillDef("C#", "language", ("csharp", "c sharp", ".net")),
    SkillDef("C", "language", (), ambiguous=True),
    SkillDef("Rust", "language", ()),
    SkillDef("Ruby", "language", ()),
    SkillDef("PHP", "language", ()),
    SkillDef("Scala", "language", ()),
    SkillDef("Kotlin", "language", ()),
    SkillDef("Swift", "language", ()),
    SkillDef("R", "language", (), ambiguous=True),
    SkillDef("SQL", "language", ("t-sql", "pl/sql", "ansi sql")),
    SkillDef("Bash", "language", ("shell scripting", "shell", "zsh")),
    SkillDef("MATLAB", "language", ()),

    # -- web / frontend ----------------------------------------------------
    SkillDef("React", "framework", ("react.js", "reactjs")),
    SkillDef("Next.js", "framework", ("nextjs",)),
    SkillDef("Vue.js", "framework", ("vue", "vuejs")),
    SkillDef("Angular", "framework", ("angularjs",)),
    SkillDef("Svelte", "framework", ("sveltekit",)),
    SkillDef("HTML", "framework", ("html5",)),
    SkillDef("CSS", "framework", ("css3", "sass", "scss", "tailwind", "tailwindcss")),
    SkillDef("Redux", "framework", ()),
    SkillDef("Node.js", "framework", ("nodejs", "node")),
    SkillDef("Express.js", "framework", ("express", "expressjs")),

    # -- backend frameworks -----------------------------------------------
    SkillDef("Django", "framework", ()),
    SkillDef("Flask", "framework", ()),
    SkillDef("FastAPI", "framework", ()),
    SkillDef("Spring Boot", "framework", ("spring",)),
    SkillDef("Ruby on Rails", "framework", ("rails",)),
    SkillDef("ASP.NET", "framework", ("asp.net core",)),
    SkillDef("GraphQL", "framework", ("apollo",)),
    SkillDef("REST APIs", "framework", ("rest", "restful", "rest api", "restful api")),
    SkillDef("gRPC", "framework", ()),
    SkillDef("Microservices", "practice", ("micro-services", "microservice architecture")),

    # -- data stores -------------------------------------------------------
    SkillDef("PostgreSQL", "database", ("postgres", "psql")),
    SkillDef("MySQL", "database", ("mariadb",)),
    SkillDef("MongoDB", "database", ("mongo",)),
    SkillDef("Redis", "database", ()),
    SkillDef("Elasticsearch", "database", ("elastic search", "opensearch", "elk")),
    SkillDef("Cassandra", "database", ()),
    SkillDef("DynamoDB", "database", ()),
    SkillDef("SQLite", "database", ()),
    SkillDef("Snowflake", "database", ()),
    SkillDef("BigQuery", "database", ()),
    SkillDef("Redshift", "database", ()),
    SkillDef("Neo4j", "database", ("graph database",)),
    SkillDef("pgvector", "database", ()),

    # -- cloud -------------------------------------------------------------
    SkillDef("AWS", "cloud", ("amazon web services",)),
    SkillDef("Azure", "cloud", ("microsoft azure",)),
    SkillDef("GCP", "cloud", ("google cloud", "google cloud platform")),
    SkillDef("Lambda", "cloud", ("aws lambda", "serverless functions")),
    SkillDef("S3", "cloud", ("aws s3",)),
    SkillDef("EC2", "cloud", ()),
    SkillDef("Cloudflare", "cloud", ("cloudflare workers",)),
    SkillDef("Serverless", "cloud", ()),

    # -- devops / platform --------------------------------------------------
    SkillDef("Docker", "devops", ("containerisation", "containerization", "containers")),
    SkillDef("Kubernetes", "devops", ("k8s", "eks", "gke", "aks")),
    SkillDef("Terraform", "devops", ("infrastructure as code", "iac")),
    SkillDef("Ansible", "devops", ()),
    SkillDef("Jenkins", "devops", ()),
    SkillDef("GitHub Actions", "devops", ("github action",)),
    SkillDef("GitLab CI", "devops", ("gitlab ci/cd",)),
    SkillDef("CI/CD", "devops", ("ci cd", "continuous integration", "continuous delivery", "continuous deployment")),
    SkillDef("Git", "devops", ("version control",)),
    SkillDef("Linux", "devops", ("unix",)),
    SkillDef("Nginx", "devops", ()),
    SkillDef("Prometheus", "devops", ()),
    SkillDef("Grafana", "devops", ()),
    SkillDef("Datadog", "devops", ()),
    SkillDef("OpenTelemetry", "devops", ("otel", "distributed tracing")),
    SkillDef("Observability", "devops", ("monitoring", "logging and monitoring")),

    # -- data engineering ---------------------------------------------------
    SkillDef("Apache Spark", "data", ("spark", "pyspark")),
    SkillDef("Apache Kafka", "data", ("kafka",)),
    SkillDef("Apache Airflow", "data", ("airflow",)),
    SkillDef("dbt", "data", ()),
    SkillDef("Databricks", "data", ()),
    SkillDef("ETL", "data", ("elt", "data pipelines", "data pipeline")),
    SkillDef("Data Modelling", "data", ("data modeling", "dimensional modelling", "star schema")),
    SkillDef("Data Warehousing", "data", ("data warehouse",)),
    SkillDef("Pandas", "data", ()),
    SkillDef("NumPy", "data", ()),
    SkillDef("Tableau", "data", ()),
    SkillDef("Power BI", "data", ("powerbi",)),
    SkillDef("Looker", "data", ()),

    # -- ML / AI ------------------------------------------------------------
    SkillDef("Machine Learning", "ml", ("ml", "supervised learning", "predictive modelling", "predictive modeling")),
    SkillDef("Deep Learning", "ml", ("neural networks",)),
    SkillDef("PyTorch", "ml", ("torch",)),
    SkillDef("TensorFlow", "ml", ("keras",)),
    SkillDef("scikit-learn", "ml", ("sklearn", "scikit learn")),
    SkillDef("NLP", "ml", ("natural language processing",)),
    SkillDef("Computer Vision", "ml", ("cv", "image processing")),
    SkillDef("LLMs", "ml", ("large language models", "large language model", "genai", "generative ai")),
    SkillDef("RAG", "ml", ("retrieval augmented generation", "retrieval-augmented generation")),
    SkillDef("Prompt Engineering", "ml", ()),
    SkillDef("LangChain", "ml", ()),
    SkillDef("LlamaIndex", "ml", ()),
    SkillDef("Vector Databases", "ml", ("vector database", "vector store", "pinecone", "weaviate", "qdrant", "chroma", "milvus", "faiss")),
    SkillDef("Hugging Face", "ml", ("huggingface", "transformers library")),
    SkillDef("MLOps", "ml", ("model deployment", "model serving")),
    SkillDef("Feature Engineering", "ml", ()),
    SkillDef("A/B Testing", "ml", ("ab testing", "experimentation", "split testing")),
    SkillDef("Statistics", "ml", ("statistical analysis", "statistical modelling")),
    SkillDef("Recommendation Systems", "ml", ("recommender systems",)),

    # -- mobile -------------------------------------------------------------
    SkillDef("iOS Development", "mobile", ("ios", "swiftui", "uikit")),
    SkillDef("Android Development", "mobile", ("android",)),
    SkillDef("React Native", "mobile", ()),
    SkillDef("Flutter", "mobile", ("dart",)),

    # -- security -----------------------------------------------------------
    SkillDef("Security", "security", ("application security", "appsec", "infosec", "cybersecurity")),
    SkillDef("OAuth", "security", ("oauth2", "openid connect", "oidc")),
    SkillDef("Penetration Testing", "security", ("pen testing", "pentesting")),
    SkillDef("Encryption", "security", ("cryptography", "tls", "ssl")),

    # -- engineering practice ------------------------------------------------
    SkillDef("Testing", "practice", ("unit testing", "unit tests", "integration testing", "test automation", "tdd", "pytest", "jest")),
    SkillDef("Code Review", "practice", ("peer review",)),
    SkillDef("System Design", "practice", ("software architecture", "distributed systems", "architecture design")),
    SkillDef("Agile", "practice", ("scrum", "kanban", "sprint planning")),
    SkillDef("Performance Optimisation", "practice", ("performance optimization", "performance tuning", "profiling")),
    SkillDef("Debugging", "practice", ("troubleshooting", "root cause analysis")),
    SkillDef("Technical Documentation", "practice", ("documentation",)),
    SkillDef("Design Patterns", "practice", ("object oriented design", "oop", "solid principles")),

    # -- product / delivery ---------------------------------------------------
    SkillDef("Product Management", "domain", ("product owner", "roadmapping", "product strategy")),
    SkillDef("Stakeholder Management", "domain", ("stakeholder engagement", "client management")),
    SkillDef("Project Management", "domain", ("delivery management", "programme management", "program management")),
    SkillDef("Requirements Gathering", "domain", ("requirement analysis", "business analysis")),
    SkillDef("Jira", "tool", ("atlassian", "confluence")),
    SkillDef("Figma", "tool", ("sketch", "wireframing", "prototyping")),
    SkillDef("Excel", "tool", ("spreadsheets", "google sheets")),

    # -- interpersonal ----------------------------------------------------------
    SkillDef("Communication", "soft", ("written communication", "verbal communication", "presentation skills")),
    SkillDef("Leadership", "soft", ("team lead", "tech lead", "leading a team", "people management")),
    SkillDef("Mentoring", "soft", ("coaching", "mentorship")),
    SkillDef("Collaboration", "soft", ("teamwork", "cross-functional", "cross functional")),
    SkillDef("Problem Solving", "soft", ("analytical thinking", "critical thinking")),
    SkillDef("Ownership", "soft", ("self-starter", "autonomy", "proactive")),
]


SKILLS: Dict[str, SkillDef] = {skill.canonical: skill for skill in _RAW}

# alias (lowercased) -> canonical name. The canonical name maps to itself.
ALIAS_TO_CANONICAL: Dict[str, str] = {}
for _skill in _RAW:
    ALIAS_TO_CANONICAL[_skill.canonical.lower()] = _skill.canonical
    for _alias in _skill.aliases:
        ALIAS_TO_CANONICAL[_alias.lower()] = _skill.canonical

AMBIGUOUS: set = {skill.canonical for skill in _RAW if skill.ambiguous}


def category_of(canonical: str) -> str:
    skill = SKILLS.get(canonical)
    return skill.category if skill else "other"
