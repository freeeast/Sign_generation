"""Matched RGB check for the learned-prior arms using the Stage 10 detector route."""
from scripts import stage10_rgb

stage10_rgb.ARMS=("A_geometry","B_stage9_joint","B_matched_expert","anchor","C_shared_only","D_shared_expert")

if __name__=="__main__":
    stage10_rgb.main()
