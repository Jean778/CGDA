# ============================================================
# Repository root (REPO) resolves automatically in Rscript and RStudio.
# Override with env var, e.g.  REPO=<repo-root>  Rscript figures/plot_x.R
# ============================================================
if (Sys.getenv("REPO") != "") {
  REPO <- Sys.getenv("REPO")
} else if (interactive()) {
  if (requireNamespace("rstudioapi", quietly = TRUE)) {
    REPO <- dirname(dirname(rstudioapi::getSourceEditorContext()$path))
  } else {
    REPO <- getwd()
  }
} else {
  .cgda_args <- commandArgs(trailingOnly = FALSE)
  .cgda_file <- sub("^--file=", "", .cgda_args[grepl("^--file=", .cgda_args)])
  REPO <- if (length(.cgda_file) > 0) dirname(dirname(.cgda_file)) else getwd()
}
library(ggplot2)
library(dplyr)
library(patchwork)

# ============================================================
# 1. 读取与筛选
# ============================================================

path <- file.path(REPO, "main_results/")
methods <- c("base", "cal_sup", "cg_sup")

s <- read.csv(
  paste0(path, "panel_44_sample_scores_cosine.csv")
) %>%
  filter(
    budget == 0.1,
    seed == 42,
    fold == 5,
    method %in% methods
  )

t <- read.csv(
  paste0(path, "panel_44_thresholds_cosine.csv")
) %>%
  filter(
    budget == 0.1,
    seed == 42,
    fold == 5,
    method %in% methods,
    threshold_method == "known_reference_q90"
  )

keys <- c(
  "protocol", "direction", "budget",
  "seed", "fold", "method", "unknown_pair"
)

stopifnot(
  nrow(s) > 0,
  !anyDuplicated(t[keys]),
  !anyDuplicated(s[c(keys, "sample_id")]),
  all(is.finite(s$distance_score)),
  all(s$distance_score >= 0),
  all(is.finite(t$threshold)),
  all(s$is_unknown %in% c(0, 1))
)

s <- left_join(
  s,
  t %>% select(all_of(keys), threshold),
  by = keys
)

stopifnot(
  !anyNA(s$threshold),
  !anyDuplicated(
    unique(s[keys])[c("method", "unknown_pair")]
  )
)

# ============================================================
# 2. 计算样本数、接受率和拒识率
# ============================================================

m <- s %>%
  group_by(method, unknown_pair) %>%
  summarise(
    tau = first(threshold),
    n_k = sum(is_unknown == 0),
    n_u = sum(is_unknown == 1),
    accept = mean(
      distance_score[is_unknown == 0] <= first(threshold)
    ),
    reject = mean(
      distance_score[is_unknown == 1] > first(threshold)
    ),
    .groups = "drop"
  ) %>%
  mutate(
    known_info = sprintf(
      "Known(n=%d)  accept:%.1f%%",
      n_k, 100 * accept
    ),
    unknown_info = sprintf(
      "Unknown(n=%d)  reject:%.1f%%",
      n_u, 100 * reject
    ),
    tau_label = sprintf("\u03C4 = %.3g", tau)
  )

stopifnot(all(m$n_k > 0), all(m$n_u > 0))

# 确认九个面板都存在
expected <- expand.grid(
  method = methods,
  unknown_pair = c("1-2", "3-5", "4-6"),
  stringsAsFactors = FALSE
)

stopifnot(
  nrow(
    anti_join(
      expected, m,
      by = c("method", "unknown_pair")
    )
  ) == 0
)

# ============================================================
# 3. 精确ECDF：合并重复分数
# ============================================================

e <- s %>%
  count(
    method, unknown_pair,
    is_unknown, distance_score
  ) %>%
  group_by(method, unknown_pair, is_unknown) %>%
  arrange(distance_score, .by_group = TRUE) %>%
  mutate(F = cumsum(n) / sum(n)) %>%
  ungroup() %>%
  mutate(
    group = ifelse(is_unknown == 0, "Known", "Unknown")
  )

# 所有子图统一横轴
x_max <- max(c(s$distance_score, m$tau))
stopifnot(x_max > 0)
x_max <- x_max * 1.04

# 右侧端点
ends <- e %>%
  distinct(method, unknown_pair, group) %>%
  mutate(
    distance_score = x_max,
    F = 1
  )

# 左侧端点；极小负值仅用于绘制零分数处的跳跃
starts <- e %>%
  group_by(method, unknown_pair, group) %>%
  summarise(
    distance_score = min(distance_score) - x_max * 1e-8,
    F = 0,
    .groups = "drop"
  )

e <- bind_rows(
  e %>%
    select(method, unknown_pair, group, distance_score, F),
  starts,
  ends
) %>%
  arrange(method, unknown_pair, group, distance_score)

# ============================================================
# 4. 配色与单个子图
# ============================================================

known_col <- "#354F80"
unknown_col <- "#DE7548"

bottom_titles <- c(
  base = "base",
  cal_sup = "cal_sup",
  cg_sup = "CGDA"
)

make_panel <- function(
    pair_value,
    method_value,
    bottom_row = FALSE
) {
  
  ee <- e %>%
    filter(
      unknown_pair == pair_value,
      method == method_value
    )
  
  mm <- m %>%
    filter(
      unknown_pair == pair_value,
      method == method_value
    )
  
  # 第二、三列标注向左上移动
  move_legend <- method_value != "base"
  
  legend_x <- x_max * if (move_legend) 0.78 else 0.98
  known_y <- if (move_legend) 0.57 else 0.20
  unknown_y <- if (move_legend) 0.46 else 0.09
  
  ggplot(
    ee,
    aes(
      x = distance_score,
      y = F,
      colour = group,
      group = group
    )
  ) +
    geom_step(
      linewidth = 0.65,
      direction = "hv"
    ) +
    
    # 阈值线
    geom_segment(
      data = mm,
      aes(
        x = tau, xend = tau,
        y = 0, yend = 1
      ),
      inherit.aes = FALSE,
      colour = "grey45",
      linetype = "dashed",
      linewidth = 0.45
    ) +
    
    # 阈值数值
    geom_text(
      data = mm,
      aes(
        x = x_max * 0.98,
        y = 1.065,
        label = tau_label
      ),
      inherit.aes = FALSE,
      hjust = 1,
      size = 2.9,
      colour = "grey30"
    ) +
    
    # Known：蓝色，与曲线一致
    geom_label(
      data = mm,
      aes(
        x = legend_x,
        y = known_y,
        label = known_info
      ),
      inherit.aes = FALSE,
      hjust = 1,
      size = 2.8,
      colour = known_col,
      fill = "white",
      linewidth = 0,
      label.padding = grid::unit(0.10, "lines")
    ) +
    
    # Unknown：橙色，与曲线一致
    geom_label(
      data = mm,
      aes(
        x = legend_x,
        y = unknown_y,
        label = unknown_info
      ),
      inherit.aes = FALSE,
      hjust = 1,
      size = 2.8,
      colour = unknown_col,
      fill = "white",
      linewidth = 0,
      label.padding = grid::unit(0.10, "lines")
    ) +
    
    scale_colour_manual(
      values = c(
        Known = known_col,
        Unknown = unknown_col
      )
    ) +
    
    scale_x_continuous(
      breaks = scales::breaks_pretty(n = 4),
      expand = expansion(mult = 0)
    ) +
    
    scale_y_continuous(
      breaks = c(0, 0.5, 1),
      labels = c("0.0", "0.5", "1.0"),
      expand = expansion(mult = 0)
    ) +
    
    coord_cartesian(
      xlim = c(0, x_max),
      ylim = c(0, 1.12)
    ) +
    
    labs(
      x = if (bottom_row) {
        unname(bottom_titles[method_value])
      } else {
        NULL
      },
      y = "Cumulative proportion"
    ) +
    
    theme_classic(
      base_size = 11,
      base_family = "sans"
    ) +
    
    theme(
      legend.position = "none",
      panel.grid = element_blank(),
      panel.border = element_blank(),
      
      axis.line = element_line(
        linewidth = 0.4,
        colour = "grey30"
      ),
      axis.ticks = element_line(
        linewidth = 0.4,
        colour = "grey30"
      ),
      axis.ticks.length = grid::unit(0.12, "cm"),
      
      axis.text = element_text(
        size = 9,
        colour = "black"
      ),
      axis.title.y = element_text(
        size = 10,
        margin = margin(r = 5)
      ),
      axis.title.x = element_text(
        size = 12,
        face = "bold",
        margin = margin(t = 8)
      ),
      
      plot.margin = margin(5, 12, 8, 5)
    )
}

# ============================================================
# 5. 每行的Unknown标题
# ============================================================

make_row_title <- function(text) {
  
  ggplot() +
    annotate(
      "text",
      x = 0.5,
      y = 0.5,
      label = text,
      size = 4,
      fontface = "bold"
    ) +
    xlim(0, 1) +
    ylim(0, 1) +
    theme_void() +
    theme(
      plot.margin = margin(0, 0, 0, 0)
    )
}

# ============================================================
# 6. 组合：每个子图均保留横纵坐标轴
# ============================================================

layout_design <- "
AAA
BCD
EEE
FGH
III
JKL
"

p <- wrap_plots(
  A = make_row_title("Unknown:1-2"),
  B = make_panel("1-2", "base"),
  C = make_panel("1-2", "cal_sup"),
  D = make_panel("1-2", "cg_sup"),
  
  E = make_row_title("Unknown:3-5"),
  F = make_panel("3-5", "base"),
  G = make_panel("3-5", "cal_sup"),
  H = make_panel("3-5", "cg_sup"),
  
  I = make_row_title("Unknown:4-6"),
  J = make_panel("4-6", "base", bottom_row = TRUE),
  K = make_panel("4-6", "cal_sup", bottom_row = TRUE),
  L = make_panel("4-6", "cg_sup", bottom_row = TRUE),
  
  design = layout_design,
  heights = c(0.16, 1, 0.16, 1, 0.16, 1)
) +
  plot_annotation(
    caption = paste0(
      "Horizontal axes: cosine distance score. ",
      "Budget 10%; seed 42; fold 5.\n",
      "Dashed lines: run-specific known-reference Q90 thresholds. ",
      "Acceptance: score <= tau; rejection: score > tau."
    ),
    theme = theme(
      plot.caption = element_text(
        size = 8,
        hjust = 0,
        margin = margin(t = 10)
      )
    )
  )

print(p)

# ============================================================
# 7. 按需保存
# ============================================================

# ggsave(
#   paste0(path, "ecdf_cosine_revised.png"),
#   p,
#   width = 11,
#   height = 9.5,
#   dpi = 300,
#   bg = "white"
# )

# ggsave(
#   paste0(path, "ecdf_cosine_revised.pdf"),
#   p,
#   width = 11,
#   height = 9.5,
#   device = grDevices::cairo_pdf
# )