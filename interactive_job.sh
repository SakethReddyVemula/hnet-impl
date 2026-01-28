#!/bin/bash
#SBATCH -J I
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p dibdp
#SBATCH -t 06:00:00
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --export=ALL,JOB_DESCRIPTION="Machine translation for Indian languages faces challenges such as rich morphology agglutination free word order and limited annotated resources This project focuses on tokenization strategies for Sanskrit Tamil translation incorporating linguistic knowledge from grammar literature vocabulary and parallel corpora Effective tokenization enables better representation of morphological units compound words and verse structure supporting accurate interpretation of ayurveda itihasa purana poetry prose anvaya philosophy and temple texts",EXPECTED_OUTCOME="The outcome is improved Sanskrit Tamil translation quality through robust tokenization methods that handle morphology compounds and long range dependencies By aligning tokens with linguistic and domain knowledge models better preserve grammatical agreement poetic structure anvaya interpretation and cultural nuance This leads to clearer more consistent translations of ayurvedic concepts historical narratives and literary texts supporting education research digital archives heritage studies and multilingual knowledge dissemination systems"

# Start screen session as required
screen -D -m
