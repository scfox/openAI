#!/bin/bash
aws s3 sync s3://openai-scf/$1/policy output/policy --profile=agilemobile
