# Phase 02 --- FastAPI, Docker, CI/CD & AWS Deployment

## Goal

Convert the Customer Churn ML model from Phase 01 into a
production-style ML application, containerize it, automate testing, and
deploy it to AWS EC2.

**Estimated duration:** 5--7 weeks

## Phase 02 Rule

> Learn → Practice → Implement in the project → Test → Milestone → Move
> forward.

This document is the **single source of truth for Phase 02**. Do not add
random courses, tutorials, technologies, or AWS services unless a
genuine project requirement appears.

---

# 1. Master Roadmap

```text
PHASE 01
Customer Churn ML Model
        |
        v
2.1 FASTAPI
Build ML API
        |
   MILESTONE 1
        |
        v
2.2 DOCKER
Containerize API
        |
   MILESTONE 2
        |
        v
DOCKER COMPOSE
FastAPI + PostgreSQL
        |
   MILESTONE 3
        |
        v
2.3 GITHUB ACTIONS
Automated CI
        |
   MILESTONE 4
        |
        v
2.4 AWS FUNDAMENTALS
IAM / EC2 / VPC / Security Groups
S3 / CloudWatch / Billing
        |
   MILESTONE 5
        |
        v
2.5 AWS DEPLOYMENT
EC2 + Docker + Nginx
S3 + CloudWatch
        |
   FINAL MILESTONE
        |
        v
PHASE 02 COMPLETE
```

---

# 2. Phase 02 Rules

## Rule 1 --- No roadmap drift

Do **not** randomly add:

- Kubernetes
- Terraform
- ECS
- EKS
- Lambda
- SageMaker
- Redis
- Kafka
- Microservices
- Advanced AWS architecture
- Advanced networking

Those belong to future learning unless a real project requirement forces
us to introduce one.

## Rule 2 --- Do not collect courses

We are not trying to finish courses.

We are trying to build:

```text
ML Model
   |
FastAPI
   |
PostgreSQL
   |
Docker
   |
Docker Compose
   |
GitHub Actions
   |
AWS EC2
   |
Nginx
   |
Public API
```

## Rule 3 --- One milestone before the next

Do not move to the next technology merely because the tutorial/course is
finished.

Move forward when the current project milestone actually works.

## Rule 4 --- Prefer official resources

Use the exact resources in this document. Do not search for another
course unless instructed.

## Rule 5 --- Free learning is preferred

Paid courses/certifications are not required for Phase 02.

---

# 3. Phase 02 Project Architecture

The final system will approximately look like:

```text
                         INTERNET
                            |
                            v
                     AWS EC2 INSTANCE
                            |
                     Security Group
                            |
                       Port 80/443
                            |
                            v
                          NGINX
                            |
                            v
                    Docker Container
                            |
                         FastAPI
                            |
                    +-------+-------+
                    |               |
                    v               v
                ML Model       PostgreSQL
                    |
                    v
              Prediction
```

Supporting AWS services:

```text
S3
 |
 +-- Customer Churn Model Backup

CloudWatch
 |
 +-- Logs
 +-- Metrics
```

---

# 4. Stage 2.1 --- FastAPI

## Objective

Turn the Phase 01 ML model into an API.

Current:

```text
Input
  |
Preprocessing
  |
ML Model
  |
Prediction
```

Target:

```text
Client
  |
FastAPI
  |
Preprocessing
  |
ML Model
  |
Prediction
  |
JSON Response
```

## Official resources

### Primary

[FastAPI Official Tutorial](https://fastapi.tiangolo.com/tutorial/)

### Database

[FastAPI SQL
Databases](https://fastapi.tiangolo.com/tutorial/sql-databases/)

## Study in this order

### FastAPI fundamentals

- First Steps
- Path Parameters
- Query Parameters
- Request Body
- Response Model
- Status Codes
- Handling Errors

### Pydantic

Learn:

- `BaseModel`
- Type hints
- Validation
- Optional fields
- Field constraints
- Nested models

### Dependency Injection

Learn:

- `Depends()`
- Dependency injection
- Reusable dependencies

### Database

Learn:

- SQLAlchemy basics
- Database connection
- Sessions
- CRUD
- PostgreSQL integration

### Documentation

Understand:

- Swagger UI
- OpenAPI
- ReDoc

## Skip for now

- WebSockets
- GraphQL
- Advanced middleware
- Advanced security
- OAuth2 deep dive
- Streaming
- Advanced deployment configurations
- Other advanced FastAPI topics not required by the project

## Project implementation

### `GET /health`

Expected concept:

```json
{
  "status": "healthy"
}
```

### `POST /predict`

Input fields must match the actual Phase 01 model features.

Example:

```json
{
  "age": 35,
  "tenure": 12,
  "monthly_charges": 79.5
}
```

Example response:

```json
{
  "prediction": 1,
  "probability": 0.82
}
```

### `POST /explain`

Return a SHAP-based model explanation.

### Prediction logging

Store:

- Request/input
- Prediction
- Probability
- Timestamp
- Model information as appropriate

## FastAPI milestone

Do not move to Docker until:

- [ ] FastAPI starts successfully
- [ ] `/health` works
- [ ] `/predict` works
- [ ] `/explain` works
- [ ] Pydantic validation works
- [ ] Errors are handled properly
- [ ] PostgreSQL is connected
- [ ] Predictions are stored
- [ ] Swagger works
- [ ] API is tested

**Milestone:** Working Customer Churn FastAPI application.

---

# 5. Stage 2.2 --- Docker

## Objective

Containerize the FastAPI application.

## Official resource

[Docker Get Started](https://docs.docker.com/get-started/)

## Study

Learn:

- Images
- Containers
- Dockerfile
- Docker build
- Docker run
- Port mapping
- Environment variables
- Volumes
- Networks

Mental model:

```text
Dockerfile
     |
     v
   Image
     |
     v
 Container
     |
     v
Running Application
```

## Project implementation

Create a:

```text
Dockerfile
```

Build the image:

```bash
docker build
```

Run the container:

```bash
docker run
```

Target:

```text
Browser
   |
localhost
   |
Docker Container
   |
FastAPI
   |
ML Model
```

## Skip for now

- Kubernetes
- Docker Swarm
- Advanced networking
- Advanced storage drivers
- Container orchestration
- Docker security deep dive

## Docker milestone

- [ ] Dockerfile created
- [ ] Image builds
- [ ] Container starts
- [ ] FastAPI is accessible from host
- [ ] Model loads inside container
- [ ] Prediction works inside container
- [ ] Environment variables work

**Milestone:** Customer Churn API runs inside Docker.

---

# 6. Stage 2.2B --- Docker Compose

## Objective

Run the complete local application with FastAPI and PostgreSQL.

## Official resources

[Docker Compose Documentation](https://docs.docker.com/compose/)

[Docker Compose
Quickstart](https://docs.docker.com/compose/gettingstarted/)

## Study

Learn:

- `compose.yaml`
- Services
- Ports
- Environment variables
- Volumes
- Networks
- `depends_on`
- Health checks

## Target architecture

```text
              Docker Compose
                    |
          +---------+---------+
          |                   |
          v                   v
      FastAPI             PostgreSQL
       Container           Container
          |                   |
          +------ Network -----+
```

Start everything with:

```bash
docker compose up --build
```

## Skip for now

- Kubernetes
- Docker Swarm
- ECS
- Advanced Compose orchestration
- Multi-region architecture

## Docker Compose milestone

- [ ] FastAPI service
- [ ] PostgreSQL service
- [ ] Network communication
- [ ] Persistent database volume
- [ ] Environment variables
- [ ] Health checks
- [ ] One-command startup

**Milestone:** Entire application runs using
`docker compose up --build`.

---

# 7. Stage 2.3 --- GitHub Actions / CI

## Objective

Automatically test the project and verify that the Docker image builds.

## Official resource

[GitHub Actions
Quickstart](https://docs.github.com/en/actions/get-started/quickstart)

## Study

Learn:

- Workflow
- Job
- Step
- Runner
- Trigger
- Action
- Secrets

Mental model:

```text
git push
   |
GitHub
   |
GitHub Actions
   |
Install dependencies
   |
Run tests
   |
Build Docker image
   |
PASS / FAIL
```

## Project implementation

Create:

```text
.github/
└── workflows/
    └── ci.yml
```

Triggers:

- Push
- Pull Request

Workflow:

1.  Checkout code
2.  Setup Python
3.  Install dependencies
4.  Run tests
5.  Build Docker image

## Skip for now

- Advanced deployment pipelines
- Kubernetes deployment
- AWS deployment through Actions
- Self-hosted runners
- Complex reusable workflows
- Matrix builds
- Advanced release automation

## CI milestone

- [ ] Workflow created
- [ ] Push triggers workflow
- [ ] Pull request triggers workflow
- [ ] Dependencies install
- [ ] Tests run
- [ ] Docker image builds
- [ ] GitHub shows success/failure

**Milestone:** Every code change automatically gets tested.

---

# 8. Stage 2.4 --- AWS Fundamentals

## AWS learning philosophy

AWS Skill Builder contains many courses and labs. Do not randomly choose
from the catalog.

Our exact AWS curriculum is:

```text
AWS-01 Fundamentals
      |
AWS-02 IAM
      |
AWS-03 EC2
      |
AWS-04 VPC
      |
AWS-05 Security Groups
      |
AWS-06 S3
      |
AWS-07 CloudWatch
      |
AWS-08 Billing
```

The AWS fundamentals course is only the map. It does **not** replace the
individual service learning below.

---

# 9. AWS-01 --- Fundamentals

## Official resource

[AWS Cloud Practitioner
Essentials](https://skillbuilder.aws/learn/94T2BEN85A/aws-cloud-practitioner-essentials/8D79F3AVR7)

**Preferred cost: Free**

## Study

Understand:

- Cloud computing
- AWS global infrastructure
- Regions
- Availability Zones
- Compute
- Storage
- Networking
- Security
- IAM
- EC2
- S3
- VPC
- CloudWatch
- Pricing
- Shared responsibility model

## Skip

Do not try to memorize:

- Every AWS service
- Certification questions
- Advanced architecture
- Advanced networking
- Advanced security
- Detailed service-specific implementation

## Goal

Be able to explain:

> What are EC2, S3, VPC, IAM and CloudWatch, and why would our ML system
> use them?

**Important:** This is an overview, not complete mastery of each AWS
service.

---

# 10. AWS-02 --- IAM

## Official resource

[AWS IAM Getting
Started](https://docs.aws.amazon.com/IAM/latest/UserGuide/getting-started.html)

**Learning cost: Free**

## Study

```text
IAM
|
+-- User
+-- Group
+-- Role
+-- Policy
+-- Permission
+-- Authentication
+-- Authorization
+-- MFA
```

Understand:

- Root account
- IAM users
- IAM roles
- IAM policies
- Permissions
- MFA
- Authentication vs authorization

## Skip

- SAML
- Federation
- ABAC
- Cross-account IAM
- AWS Organizations
- SCPs
- Advanced policy conditions

## Goal

Understand:

> Who can access AWS and what are they allowed to do?

---

# 11. AWS-03 --- EC2

## Official resources

[Amazon EC2 Getting
Started](https://aws.amazon.com/ec2/getting-started/)

[EC2 Getting Started
Documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html)

**Learning resources: Free**

Actual AWS usage may incur charges depending on account status, offers,
and resource usage.

## Study

- EC2 instance
- AMI
- Instance types
- Key pair
- EBS
- Public IP
- Private IP
- SSH
- Linux server
- Security Group

Mental model:

```text
AMI
 |
 v
EC2 Instance
 |
 v
Linux
 |
 v
Docker
 |
 v
FastAPI
```

## Practical concepts

Learn to:

1.  Launch an EC2 instance
2.  Select an AMI
3.  Select an instance type
4.  Configure a key pair
5.  Configure a security group
6.  Connect using SSH
7.  Stop an instance
8.  Terminate an instance

## Skip

- Auto Scaling
- Spot Fleet
- Placement Groups
- Dedicated Hosts
- HPC
- Advanced EBS
- Advanced AMI management

## Goal

Understand:

> EC2 is our cloud server where Docker + FastAPI will eventually run.

---

# 12. AWS-04 --- VPC

## Official resource

[AWS VPC
Documentation](https://docs.aws.amazon.com/vpc/latest/userguide/what-is-amazon-vpc.html)

**Free**

## Study

```text
VPC
|
+-- Subnet
+-- Route Table
+-- Internet Gateway
+-- Security Group
```

Understand:

- VPC
- Public subnet
- Basic routing
- Internet Gateway
- How EC2 connects to the internet

## Skip

- Transit Gateway
- VPC Peering
- PrivateLink
- VPN
- Direct Connect
- Network Firewall
- Advanced routing
- Complex enterprise networking

## Goal

Understand:

```text
Internet
   |
Internet Gateway
   |
VPC
   |
Subnet
   |
EC2
   |
Security Group
```

---

# 13. AWS-05 --- Security Groups

## Official resource

[Security Groups for your
VPC](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-security-groups.html)

**Free**

## Study

- Inbound rules
- Outbound rules
- Ports
- Protocols
- Sources

For our application:

```text
22  -> SSH
80  -> HTTP
443 -> HTTPS
```

## Skip

- Advanced Network ACL design
- AWS Network Firewall
- Firewall Manager
- Enterprise security architecture

## Goal

Understand why different ports and sources require different security
rules.

---

# 14. AWS-06 --- S3

## Official resources

[Amazon S3 Getting Started](https://aws.amazon.com/s3/getting-started/)

[S3 Getting Started
Documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/GetStartedWithS3.html)

**Learning: Free**

## Study

```text
S3
|
+-- Bucket
+-- Object
+-- Region
+-- Upload
+-- Download
+-- Permissions
+-- Versioning
```

Project use:

```text
S3 Bucket
|
+-- models/
      |
      +-- customer_churn_model.joblib
```

## Skip

- Glacier
- Replication
- Transfer Acceleration
- Object Lambda
- Access Points
- Advanced lifecycle architecture

## Goal

Understand how to create/use a bucket, upload a model backup, understand
permissions, retrieve it, and clean it up.

---

# 15. AWS-07 --- CloudWatch

## Official resources

[CloudWatch Getting
Started](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/GettingStarted.html)

[CloudWatch Logs Getting
Started](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/CWL_GettingStarted.html)

**Learning: Free**

## Study

```text
CloudWatch
|
+-- Metrics
+-- Logs
+-- Log Groups
+-- Log Streams
+-- Alarms
```

Project architecture:

```text
EC2
|
+-- FastAPI logs
+-- Application logs
+-- System metrics
       |
       v
   CloudWatch
```

## Skip

- Advanced dashboards
- Contributor Insights
- Synthetics
- Application Signals
- X-Ray
- Advanced observability architecture

## Goal

Understand:

> CloudWatch lets us monitor AWS resources and collect/view logs and
> metrics.

---

# 16. AWS-08 --- Billing & Cost Management

## Official resource

[AWS Billing and Cost
Management](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/billing-what-is.html)

**Free**

## Study

- Billing
- Cost Explorer
- Budgets
- Alerts
- Free Tier/free offerings
- Usage
- Resource cleanup

Mental model:

```text
AWS Resource
     |
   Usage
     |
    Cost
     |
  Billing
```

## Skip

- Enterprise billing
- Complex chargeback
- Organizations billing architecture
- FinOps architecture

## Goal

Know how to:

- See usage/cost
- Create a budget/alert
- Identify resources
- Shut down unnecessary resources
- Clean up after the project

---

# 17. Optional AWS Builder Labs

AWS has hands-on Builder Labs covering introductory services such as
IAM, EC2, VPC, and S3.

Use them **only if they are available to you for free**.

Useful labs:

- Introduction to IAM
- Introduction to EC2
- Introduction to VPC
- Introduction to S3

If AWS asks you to upgrade/pay for Skill Builder:

> **Do not pay just for these labs.**

Use the official documentation and our actual project deployment
instead.

---

# 18. Stage 2.5 --- AWS Deployment

Now we stop studying AWS theoretically and build.

---

## Deployment 1 --- EC2

Create:

```text
AWS
 |
EC2
 |
Linux
```

Connect:

```text
Your PC
 |
SSH
 |
EC2
```

---

## Deployment 2 --- Install Docker

On EC2:

```text
EC2
 |
Docker
 |
Docker Compose
```

---

## Deployment 3 --- Get Project

```text
GitHub
 |
EC2
 |
Customer Churn Project
```

---

## Deployment 4 --- Start Application

```bash
docker compose up --build
```

Architecture:

```text
EC2
 |
Docker Compose
 |
+-- FastAPI
+-- PostgreSQL
```

---

## Deployment 5 --- Nginx

Target:

```text
Internet
   |
Nginx
   |
FastAPI
```

Nginx is the reverse proxy.

## Learn

- Installation
- Configuration
- Reverse proxy
- Port forwarding
- Basic server block
- HTTP
- HTTPS/SSL if required for the final demonstration

No separate large Nginx course is required.

---

## Deployment 6 --- Public API

Target:

```text
Internet
    |
EC2 Public IP
    |
Nginx :80
    |
FastAPI Docker Container
    |
ML Model
```

Verify externally:

```text
GET  /health
POST /predict
POST /explain
```

---

## Deployment 7 --- S3 Model Backup

Upload:

```text
customer_churn_model.joblib
```

to:

```text
S3
|
+-- models/
      |
      +-- customer_churn_model.joblib
```

---

## Deployment 8 --- CloudWatch

Configure basic monitoring/logging.

Target:

```text
EC2
|
+-- System metrics
+-- Application logs
       |
       v
  CloudWatch
```

---

# 19. Final System Architecture

```text
                         +--------------+
                         |   INTERNET   |
                         +------+-------+
                                |
                                v
                         +--------------+
                         |    AWS EC2   |
                         |              |
                         | Security     |
                         | Group        |
                         +------+-------+
                                |
                                v
                         +--------------+
                         |    NGINX     |
                         | Reverse Proxy|
                         +------+-------+
                                |
                                v
                 +---------------------------+
                 |      DOCKER COMPOSE       |
                 |                           |
                 | +-----------------------+ |
                 | |       FastAPI         | |
                 | |                       | |
                 | | /health               | |
                 | | /predict              | |
                 | | /explain              | |
                 | +----------+------------+ |
                 |            |              |
                 | +----------v------------+ |
                 | |      PostgreSQL       | |
                 | +-----------------------+ |
                 +---------------------------+
                              |
                     +--------+--------+
                     |                 |
                     v                 v
                    S3             CloudWatch
              Model Backup       Logs / Metrics
```

---

# 20. Final Deliverables

## Application

- [ ] Production-ready FastAPI application
- [ ] `/health`
- [ ] `/predict`
- [ ] `/explain`
- [ ] Pydantic validation
- [ ] Error handling
- [ ] PostgreSQL integration
- [ ] Prediction logging

## Docker

- [ ] Dockerfile
- [ ] Docker image
- [ ] Docker container
- [ ] Docker Compose
- [ ] PostgreSQL container
- [ ] Persistent volume
- [ ] Environment variables
- [ ] Service networking

## CI

- [ ] GitHub Actions
- [ ] Automated tests
- [ ] Docker build
- [ ] Push trigger
- [ ] Pull-request trigger

## AWS

- [ ] IAM
- [ ] EC2
- [ ] VPC basics
- [ ] Security Groups
- [ ] S3
- [ ] CloudWatch
- [ ] Billing/cost awareness

## Deployment

- [ ] EC2 server
- [ ] Docker installed
- [ ] Application running
- [ ] Nginx
- [ ] Public API
- [ ] S3 model backup
- [ ] CloudWatch logs/monitoring

---

# 21. Phase 02 Completion Criteria

Phase 02 is complete only when all of these are true:

- [ ] FastAPI API works
- [ ] `/predict` works
- [ ] `/health` works
- [ ] `/explain` works
- [ ] PostgreSQL connected
- [ ] Prediction history stored
- [ ] Docker works
- [ ] Docker Compose works
- [ ] GitHub Actions works
- [ ] Tests automatically run
- [ ] Docker image builds in CI
- [ ] IAM understood/configured
- [ ] EC2 launched
- [ ] VPC basics understood
- [ ] Security Group configured
- [ ] S3 model backup exists
- [ ] CloudWatch logging works
- [ ] Billing/cost monitoring understood/configured as appropriate
- [ ] Nginx configured
- [ ] API accessible publicly
- [ ] End-to-end prediction verified

Then:

# PHASE 02 COMPLETE

Only after this do we move to:

# PHASE 03 --- Deep Learning & PyTorch

---

# 22. Exact Execution Order

This is the part to follow day-to-day.

```text
1. FASTAPI
   |
   +-- Official Tutorial
   +-- Learn required sections
   +-- Build API
   +-- Connect PostgreSQL
   +-- Add /health
   +-- Add /predict
   +-- Add /explain
   +-- Test
   |
   v
2. FASTAPI MILESTONE
   |
   v
3. DOCKER
   |
   +-- Dockerfile
   +-- Build
   +-- Run
   +-- Test
   |
   v
4. DOCKER MILESTONE
   |
   v
5. DOCKER COMPOSE
   |
   +-- FastAPI
   +-- PostgreSQL
   +-- Network
   +-- Volume
   +-- Environment
   +-- Healthcheck
   |
   v
6. COMPOSE MILESTONE
   |
   v
7. GITHUB ACTIONS
   |
   +-- Tests
   +-- Docker build
   +-- Push
   +-- Pull Request
   |
   v
8. CI MILESTONE
   |
   v
9. AWS FUNDAMENTALS
   |
   +-- Cloud Practitioner Essentials
   |
   v
10. IAM
    |
    v
11. EC2
    |
    v
12. VPC
    |
    v
13. SECURITY GROUPS
    |
    v
14. S3
    |
    v
15. CLOUDWATCH
    |
    v
16. BILLING
    |
    v
17. AWS DEPLOYMENT
    |
    +-- EC2
    +-- Docker
    +-- Docker Compose
    +-- Nginx
    +-- Public API
    +-- S3 backup
    +-- CloudWatch
    |
    v
18. FINAL END-TO-END TEST
    |
    v
19. PHASE 02 COMPLETE
```

---

# 23. Mentor Workflow

For every stage, use this process:

```text
LEARN
  |
  v
PRACTICE
  |
  v
IMPLEMENT
  |
  v
TEST
  |
  v
FIX
  |
  v
MILESTONE
  |
  v
NEXT STAGE
```

When you encounter an error, do not randomly change the architecture or
jump to another tutorial.

Bring:

1.  The relevant code
2.  The exact error
3.  What you expected
4.  What actually happened

We diagnose it, fix it, verify it, and continue.

---

# 24. Starting Point

Do **not** start AWS yet.

Do **not** start Docker yet.

Start here:

## FastAPI

[FastAPI Official Tutorial](https://fastapi.tiangolo.com/tutorial/)

Study in this order:

```text
First Steps
    ↓
Path Parameters
    ↓
Query Parameters
    ↓
Request Body
    ↓
Response Model
    ↓
Status Codes
    ↓
Handling Errors
```

Then implement the first FastAPI exercise in the Customer Churn project.

**Do not move ahead to Docker until the FastAPI milestone is complete.**

---

# Phase 02 Status

**Current stage:** 2.1 FastAPI

**Next milestone:** Working Customer Churn FastAPI application

**Current action:** Study the required FastAPI tutorial sections and
then implement them in the project.

**Roadmap status:** LOCKED --- do not drift.
