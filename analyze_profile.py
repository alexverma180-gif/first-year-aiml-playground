
import pstats
stats = pstats.Stats("profile.prof")
stats.sort_stats("cumulative").print_stats(20)
