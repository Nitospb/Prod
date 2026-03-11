# PROD 2026 - Look-a-Like

## Important

Russian version: [README.md](README.md)

> [!TIP]
> The assignment can be found in a separate repository at the link – https://gitlab.prodcontest.com/2026-final-tasks/ml


Use this repository as a solution structure template.

## What is considered your submission

The evaluated submission is the latest commit in the `main` branch.

## Minimum required repository structure

The repository must include:

1. `Dockerfile`
2. `docker-compose.yml`
3. `dvc.yaml`
4. `params.yaml`

These files are validated by the sample `.gitlab-ci.yml` in this template.

## Runtime expectations

During evaluation, after `docker compose up -d`, your service must:

1. listen on port `80`;
2. return `200` on `GET /ready` within 180 seconds;
3. implement endpoints according to `openapi.yml`.

## Academic integrity and communication culture

We urge all participants to adhere to the principles of academic integrity and a culture of communication, and to approach competitions openly and in good faith. <br/>
The purpose of the Olympiad is not only to demonstrate your knowledge and skills, but also to develop as reliable and responsible specialists in the future.

### Academic integrity and loan screening

We check the independence of the solutions:

- internal verification of the organizers;
- external verification via Codechecker (an Anti-Plagiat company product).

You can use code from open sources, but it is important to specify the source in the comment next to the fragment or in the `README'. 

If you use LLM/neural networks (for example, to generate code snippets, tests, or documentation), please mark this, where appropriate, in a comment next to the fragment or a separate note in the `README`. <br/>
Important: **you are responsible for all the code in the repository**, including the generated fragments — you must understand exactly what you are adding, be able to explain the solution and check the result (correctness, security, extreme cases).

This will make it easy to check and confirm the independence of your work and remove possible issues during verification.

### Communication culture and ethics of participation

The Industrial Programming Olympiad is about a professional approach. Therefore, we expect a respectful and businesslike tone in the repository (commit messages, branch names, code comments, and discussions).

We do not accept profanity, insults and toxic behavior.

## Measures in case of violations

If we find a violation of academic integrity or a culture of communication, we can cancel the result of the work (including disqualification from the Olympiad). <br/>
The decision is made by the organizers based on the totality of signs and results of inspections.
