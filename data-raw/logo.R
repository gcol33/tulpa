# Gaussian curves use the observed sepal-length mean and SD of each iris species.
# Each curve and its shadow move together on a separate loop.
paper <- "#F5F6F8"
colours <- c("#2A7A7A", "#7A4F9E", "#2E6F9E")
groups <- split(datasets::iris$Sepal.Length, datasets::iris$Species)
x <- seq(2.8, 9.2, length.out = 121)
px <- (x - 3.2) / 5.6 * 520
hex <- function(radius) {
  angle <- (seq_len(6) - 1) * pi / 3 - pi / 2
  paste(sprintf("%.2f,%.2f", 260 + radius * cos(angle),
                300 + radius * sin(angle)), collapse = " ")
}
curve_path <- function(i, phase = 0, close = FALSE) {
  mu <- mean(groups[[i]]) + 0.22 * sin(phase)
  baseline <- 325 + (i - 1) * 27
  yy <- baseline - 175 * dnorm(x, mu, sd(groups[[i]]))
  path <- paste("M", paste(sprintf("%.2f,%.2f", px, yy), collapse = " L "))
  if (close) path <- paste(path, sprintf("L %.2f,%.2f L %.2f,%.2f Z",
                                       tail(px, 1), baseline + 12,
                                       px[1], baseline + 12))
  path
}
species_layer <- function(i, animated, outline = FALSE) {
  id <- paste0("tulpa-shadow-", if (animated) "moving" else "still", "-", i)
  drift <- if (animated) sprintf(
    '<animateTransform attributeName="transform" type="translate" values="-9 0;9 0;-9 0" begin="0s" dur="%ss" calcMode="linear" repeatCount="indefinite"/>',
    c(8, 11, 14)[i]) else ""
  if (outline) return(sprintf('<g class="tulpa-%s">
<path d="%s" fill="none" stroke="%s" stroke-width="6" stroke-linecap="round"/>
%s</g>', if (animated) "moving" else "still", curve_path(i), colours[i], drift))
  shadows <- paste(sprintf('<use href="#%s" transform="translate(0 %s)"/>',
                           id, c(-95, -65, -35)), collapse = "\n")
  sprintf('<g class="tulpa-%s">
<defs><path id="%s" d="%s" fill="none" stroke="%s" stroke-width="2.5" opacity="0.23"/></defs>
%s
<path d="%s" fill="%s" fill-opacity="0.22" filter="url(#tulpa-area-shadow)"/>
<path d="%s" fill="%s" fill-opacity="0.13"/>
%s</g>', if (animated) "moving" else "still", id,
          curve_path(i), colours[i], shadows,
          curve_path(i, close = TRUE), colours[i],
          curve_path(i, close = TRUE), colours[i],
          drift)
}
svg <- sprintf('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 600" role="img" aria-label="tulpa logo">
<title>tulpa logo</title>
<defs><clipPath id="tulpa-hex"><polygon points="%s"/></clipPath>
<filter id="tulpa-area-shadow" x="-10%%" y="-10%%" width="120%%" height="120%%"><feGaussianBlur stdDeviation="5"/></filter></defs>
<style>.tulpa-still{display:none}@media(prefers-reduced-motion:reduce){.tulpa-moving{display:none}.tulpa-still{display:inline}}</style>
<polygon points="%s" fill="%s"/>
<g clip-path="url(#tulpa-hex)">
%s
</g>
<text x="260" y="472" text-anchor="middle" font-family="Georgia, &apos;Times New Roman&apos;, serif" font-size="78" font-weight="bold" font-style="italic" fill="#2E6F9E" stroke="%s" stroke-width="10" paint-order="stroke">tulpa</text>
<polygon points="%s" fill="none" stroke="#2B3A4A" stroke-width="18" stroke-linejoin="round"/>
</svg>', hex(288), hex(288), paper,
paste(unlist(lapply(c(FALSE, TRUE), function(outline)
  vapply(seq_along(groups), function(i)
    paste(species_layer(i, TRUE, outline), species_layer(i, FALSE, outline),
          sep = "\n"), character(1)))), collapse = "\n"), paper, hex(291))
stopifnot(!grepl("Inf|NaN", svg), nchar(svg, type = "bytes") < 200000)
dir.create("man/figures", recursive = TRUE, showWarnings = FALSE)
writeLines(svg, "man/figures/logo.svg", useBytes = TRUE)
