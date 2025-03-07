#!/bin/bash
# Script to remove large files from Git history
# Created for MLX LLM Tutorial project

# Set text colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Git Large File Removal Tool ===${NC}"
echo -e "${YELLOW}This script will help remove the Miniconda installer from your Git history${NC}"
echo -e "${YELLOW}Make sure you have committed all your changes before proceeding${NC}\n"

# Check if git-filter-repo is installed
if ! command -v git-filter-repo &> /dev/null; then
    echo -e "${RED}git-filter-repo is not installed.${NC}"
    echo -e "${YELLOW}Would you like to install it now? (y/n)${NC}"
    read -r install_choice
    
    if [[ $install_choice == "y" || $install_choice == "Y" ]]; then
        echo -e "${GREEN}Installing git-filter-repo...${NC}"
        if command -v pip &> /dev/null; then
            pip install git-filter-repo
        elif command -v pip3 &> /dev/null; then
            pip3 install git-filter-repo
        elif command -v brew &> /dev/null; then
            brew install git-filter-repo
        else
            echo -e "${RED}Could not find pip, pip3, or brew to install git-filter-repo.${NC}"
            echo -e "${YELLOW}Please install git-filter-repo manually and run this script again.${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Please install git-filter-repo manually and run this script again.${NC}"
        echo -e "${BLUE}You can install it with:${NC}"
        echo "  pip install git-filter-repo"
        echo "  or"
        echo "  brew install git-filter-repo (on macOS with Homebrew)"
        exit 1
    fi
fi

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree &> /dev/null; then
    echo -e "${RED}Error: Not in a git repository.${NC}"
    echo -e "${YELLOW}Please run this script from within your git repository.${NC}"
    exit 1
fi

# Confirm with the user
echo -e "${YELLOW}This will permanently remove the Miniconda installer file from your Git history.${NC}"
echo -e "${RED}WARNING: This operation is destructive and cannot be undone!${NC}"
echo -e "${YELLOW}It will rewrite your Git history, which will require a force push.${NC}"
echo -e "${YELLOW}Are you sure you want to continue? (y/n)${NC}"
read -r choice

if [[ $choice != "y" && $choice != "Y" ]]; then
    echo -e "${BLUE}Operation cancelled.${NC}"
    exit 0
fi

# Create a backup branch just in case
current_branch=$(git symbolic-ref --short HEAD)
backup_branch="${current_branch}_backup_$(date +%Y%m%d%H%M%S)"
echo -e "${BLUE}Creating backup branch: ${backup_branch}${NC}"
git branch "$backup_branch"

# Remove the large file from history
echo -e "${BLUE}Removing Miniconda3-latest-MacOSX-arm64.sh from Git history...${NC}"
echo -e "${YELLOW}This may take a while...${NC}"
git filter-repo --path Miniconda3-latest-MacOSX-arm64.sh --invert-paths

# Verify the file is gone
echo -e "${GREEN}File removal complete!${NC}"
echo -e "${BLUE}Checking if the file is still in the repository...${NC}"
if git ls-files | grep -q "Miniconda3-latest-MacOSX-arm64.sh"; then
    echo -e "${RED}Warning: The file is still in your working directory.${NC}"
    echo -e "${YELLOW}It has been removed from history but is still present locally.${NC}"
    echo -e "${YELLOW}You should delete it manually and commit the change.${NC}"
else
    echo -e "${GREEN}The file is no longer tracked by Git.${NC}"
fi

# Instructions for pushing
echo -e "\n${BLUE}=== Next Steps ===${NC}"
echo -e "${YELLOW}To push these changes to GitHub, you'll need to force push:${NC}"
echo -e "${GREEN}git push origin $current_branch --force${NC}"
echo -e "${YELLOW}Note: This will overwrite the remote history. Make sure your team is aware of this change.${NC}"
echo -e "${BLUE}If anything went wrong, you can restore from the backup branch:${NC}"
echo -e "${GREEN}git checkout $backup_branch${NC}"
echo -e "${GREEN}git branch -D $current_branch${NC}"
echo -e "${GREEN}git checkout -b $current_branch${NC}"

echo -e "\n${BLUE}=== Preventing Future Issues ===${NC}"
echo -e "${YELLOW}1. The .gitignore file has been updated to exclude installer files${NC}"
echo -e "${YELLOW}2. Always use the dedicated ~/ai-training directory for your projects${NC}"
echo -e "${YELLOW}3. Use the one-step installation method for Miniconda:${NC}"
echo -e "${GREEN}   curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh | bash${NC}"

echo -e "\n${GREEN}Done!${NC}"