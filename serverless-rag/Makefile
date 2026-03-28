TEMPLATE := template.yaml
# Defaults — override on the command line
Q ?= What is Amazon Aurora?
K ?= 5
USER_ID ?= test
DOC ?= doc-001
VENV := ../.venv/bin/activate
STACK := serverless-rag
BACKEND ?= aurora
CLUSTER_ID := serverless-rag-db
REGION ?= us-east-1
ENV_FILE := ../.env

-include $(ENV_FILE)
export

.PHONY: lint deploy deploy-guided delete clean invoke setup-local bench ingest seed answer compile

lint:
	@echo "==> Formatting Makefile..."
	mbake format Makefile
	@echo "==> Validating Makefile..."
	mbake validate Makefile
	@echo "==> Linting CloudFormation template..."
	source $(VENV) && cfn-lint $(TEMPLATE)
	@echo "==> Running mypy..."
	source $(VENV) && python -m mypy src/ --ignore-missing-imports
	@echo "==> Running pydocstyle..."
	source $(VENV) && python -m pydocstyle src/
	@echo "==> All checks passed."

compile:
	source $(VENV) && pip-compile src/requirements.txt --output-file src/requirements.lock

deploy:
	sam build && sam deploy

deploy-guided:
	sam build && sam deploy --guided

delete:
	sam delete

# Tear down everything — Aurora, S3 model bucket, SSM params, SAM stack
clean:
	@echo "==> Deleting SAM stack..."
	-sam delete --no-prompts
	@echo "==> Deleting Aurora instance(s)..."
	-for id in $$(aws rds describe-db-instances \
	--filters "Name=db-cluster-id,Values=$(CLUSTER_ID)" \
	--query 'DBInstances[*].DBInstanceIdentifier' \
	--output text --region $(REGION) 2>/dev/null); do \
	aws rds delete-db-instance --db-instance-identifier $$id \
	--skip-final-snapshot --region $(REGION); \
	done
	@echo "==> Waiting for instance(s) to be deleted..."
	-aws rds wait db-instance-deleted \
	    --filters "Name=db-cluster-id,Values=$(CLUSTER_ID)" \
	    --region $(REGION)
	@echo "==> Deleting Aurora cluster (no final snapshot)..."
	-aws rds delete-db-cluster \
	    --db-cluster-identifier $(CLUSTER_ID) \
	    --skip-final-snapshot \
	    --region $(REGION)
	@echo "==> Emptying and deleting S3 model bucket..."
	-MODEL_BUCKET=$$(aws ssm get-parameter \
	    --name /serverless-rag/model-bucket \
	    --query 'Parameter.Value' --output text --region $(REGION) 2>/dev/null); \
	    aws s3 rm s3://$$MODEL_BUCKET --recursive; \
	    aws s3 rb s3://$$MODEL_BUCKET
	@echo "==> Deleting SSM parameters..."
	-aws ssm delete-parameter --name /serverless-rag/aurora-endpoint --region $(REGION)
	-aws ssm delete-parameter --name /serverless-rag/model-bucket    --region $(REGION)
	@echo "==> Done. All resources removed."

setup-local:
	bash local/setup.sh

API_URL_CMD = aws cloudformation describe-stacks \
	--stack-name $(STACK) \
	--query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
	--output text --region $(REGION)

# make bench Q="What is Aurora?"
bench:
	@API_URL=$$($(API_URL_CMD)); \
	curl -s -G "$$API_URL/benchmark" \
	--data-urlencode "q=$(Q)" \
	--data-urlencode "user_id=$(USER_ID)" \
	-d "k=$(K)" \
	| python3 -m json.tool

# Seed all .txt files in seeds/ into both backends
seed:
	@API_URL=$$($(API_URL_CMD)); \
	python3 seeds/seed.py "$$API_URL" $(USER_ID)

# make answer Q="What is express configuration?" BACKEND=aurora
answer:
	@API_URL=$$($(API_URL_CMD)); \
	curl -s -G "$$API_URL/answer" \
	--data-urlencode "q=$(Q)" \
	--data-urlencode "user_id=$(USER_ID)" \
	-d "backend=$(BACKEND)&k=$(K)" \
	| python3 -m json.tool

# make ingest DOC=doc-001 TEXT="Amazon Aurora is..."
ingest:
	@API_URL=$$($(API_URL_CMD)); \
	curl -s -X POST "$$API_URL/ingest" \
	-H "Content-Type: application/json" \
	-d '{"user_id":"$(USER_ID)","document_id":"$(DOC)","text":"$(TEXT)"}' \
	| python3 -m json.tool

# Usage:
#   make invoke FUNCTION=IngestFunction EVENT=ingest-event.json LOCAL=1
#   make invoke FUNCTION=IngestFunction EVENT=ingest-event.json
invoke:
ifeq ($(LOCAL),1)
	sam local invoke $(FUNCTION) \
	    --event local/$(EVENT) \
	    --env-vars local/env_variables.json
else
	aws lambda invoke \
	    --function-name $(STACK)-$(FUNCTION) \
	    --payload file://local/$(EVENT) \
	    /tmp/response.json && cat /tmp/response.json
endif