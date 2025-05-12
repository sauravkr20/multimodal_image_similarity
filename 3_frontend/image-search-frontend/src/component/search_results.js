import React, { useState, useMemo } from "react";
import {
  Grid,
  Card,
  CardMedia,
  Box,
  Typography,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from "@mui/material";
import ProductModal from "./product_modal";

const IMAGE_BASE_URL = "http://localhost:5000/images/";

export default function SearchResults({ results, method }) {
  const [open, setOpen] = useState(false);
  const [selectedItemId, setSelectedItemId] = useState(null);

  const [visibleCount, setVisibleCount] = useState(10);

  // Sorting state (only relevant for clip_gemini_chroma)
  const [sortKey, setSortKey] = useState("combined_score");

  const handleOpen = (itemId) => {
    setSelectedItemId(itemId);
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
    setSelectedItemId(null);
  };

  const handleLoadMore = () => {
    setVisibleCount((prev) => Math.min(prev + 10, results.length));
  };

  const handleSortChange = (event) => {
    setSortKey(event.target.value);
    setVisibleCount(10); // Reset visible count on sort change
  };

  // Memoize sorted results to avoid unnecessary re-sorts
  const sortedResults = useMemo(() => {
    if (method === "clip_gemini_chroma") {
      // Defensive check: ensure sortKey exists on items
      return [...results].sort((a, b) => {
        const aVal = a[sortKey] ?? 0;
        const bVal = b[sortKey] ?? 0;
        return bVal - aVal; // Descending order
      });
    }
    // For other methods, no sorting applied
    return results;
  }, [results, sortKey, method]);

  const visibleResults = sortedResults.slice(0, visibleCount);

  return (
    <>
      {results.length > 0 && <Typography variant="subtitle1">Responses:</Typography>}

      {/* Sort selector for clip_gemini_chroma */}
      {method === "clip_gemini_chroma" && (
        <Box sx={{ mb: 2, maxWidth: 300 }}>
          <FormControl fullWidth size="small">
            <InputLabel id="sort-select-label">Sort By</InputLabel>
            <Select
              labelId="sort-select-label"
              id="sort-select"
              value={sortKey}
              label="Sort By"
              onChange={handleSortChange}
            >
              <MenuItem value="combined_score">Combined Score</MenuItem>
              <MenuItem value="image_score">Image Score</MenuItem>
              <MenuItem value="text_score">Text Score</MenuItem>
            </Select>
          </FormControl>
        </Box>
      )}

      <Grid container spacing={3}>
        {visibleResults.map((item, idx) => {
          const img = item;
          return (
            <Grid item xs={12} sm={6} md={4} key={idx}>
              <Card
                sx={{ position: "relative", cursor: "pointer" }}
                onClick={() => handleOpen(img.item_id)}
              >
                <CardMedia
                  component="img"
                  height="200"
                  image={IMAGE_BASE_URL + img.image_path}
                  alt={`Result ${idx + 1}`}
                />
                <Box
                  sx={{
                    bottom: 0,
                    left: 0,
                    width: "100%",
                    bgcolor: "rgba(0,0,0,0.6)",
                    color: "#fff",
                    p: 1,
                    display: "flex",
                    flexDirection: "column",
                  }}
                >
                  <Typography variant="body2">
                        <strong>Combined Score:</strong> {img.combined_score.toFixed(4)}
                      </Typography>
                  <Typography variant="body2">
                    <strong>Item ID:</strong> {img.item_id}
                  </Typography>
                  {method === "clip_gemini_chroma" && (
                    <>
                      <Typography variant="body2">
                        <strong>Combined Score:</strong> {img.combined_score.toFixed(4)}
                      </Typography>
                      <Typography variant="body2">
                        <strong>Image Score:</strong> {img.image_score.toFixed(4)}
                      </Typography>
                      <Typography variant="body2">
                        <strong>Text Score:</strong> {img.text_score.toFixed(4)}
                      </Typography>
                    </>
                  )}

                  {method === "clip_chroma" && (
                    <>
                      <Typography variant="body2">
                        <strong>Score:</strong> {img.score.toFixed(4)}
                      </Typography>
                    </>
                  )}
                </Box>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {/* Load More button */}
      {visibleCount < results.length && (
        <Box sx={{ textAlign: "center", mt: 3 }}>
          <Button variant="contained" onClick={handleLoadMore}>
            Load More
          </Button>
        </Box>
      )}

      {selectedItemId && (
        <ProductModal open={open} onClose={handleClose} itemId={selectedItemId} />
      )}
    </>
  );
}
