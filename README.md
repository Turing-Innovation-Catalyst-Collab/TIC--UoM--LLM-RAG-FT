# TIC-Repository-Template
This is a template repository to be used as the base for all new repos created for TIC projects. It comes with a set of default issues types and settings to ensure a standardised setup compliant with good practice recommended by the University of Manchester Research Software Engineering department. It is based on the [RSE department template](https://github.com/UoMResearchIT/RSE-Repository-Template).

# How to use the template when creating a new repo
1. Browse to the [TIC github project](https://github.com/Turing-Innovation-Catalyst-Collab) and click `New`
2. Under `Template` select this template
3. Ensure the `Include all branches` box is ***not*** ticked
4. Choose an appropriate repo name and description and set the repo to `Private`

# Immediate Set Up
The following steps will need to be performed immediately after creating the new repository to complete the set up:

## 1. Create main branch
You will see that `development` is the default branch. Click on the triangle next to the world `development` and type `main` in the search bar, then click `create main from development`. This then creates the `main` branch, which is used for releases, whereas the `development` branch is used for development work in progress.

## 2. Branch Protection Ruleset
Next, copy the branch protection rules from the template. Download the [branch protection rules JSON](https://livemanchesterac.sharepoint.com/:u:/s/UOM-FSE-TIC/EeO8z6n0xFlKhfGKXEAzmXMBQC8vgsM5EzrkPgBsrEUrKA?e=3bbqlb) from the UOM-FSE-TIC sharepoint.
To import these rules to your new repo, browse to your new repo and go to `Settings -> Rules -> Rulesets`, then choose `Import Ruleset` and import from the JSON file you just downloaded.

These branch protection rules are designed for a gitflow workflow. For further details see [here](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).
