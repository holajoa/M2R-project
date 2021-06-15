library(spatstat)
library(sf)
library(dplyr)
library(maptools)
library(ggmap)
library(ggthemes)
library(stringr)

myFiles <- list.files(pattern="[20]*-*-city-of-london-street.csv")
dfs <- vector(mode = "list", length = 13)

for (i in 1:13) {
  crime.data <- read.csv(myFiles[i])
  burglary <- crime.data[crime.data$Crime.type %in% c("Burglary"),]
  location <- burglary[c("Longitude", "Latitude")]
  location <- location[complete.cases(location),]
  dfs[[i]] <- location
}

location.df <- do.call("rbind", dfs)
location.sf <- st_as_sf(location.df,
                        coords = c("Longitude", "Latitude"),
                        crs = 4326)
location.sf <- st_transform(location.sf, 27700)
location.sp <- as(location.sf, "Spatial")
location.ppp <- as(location.sp, "ppp")
location.ppp <- rescale(location.ppp, 1000)
# --------------- 
london.map <- get_stamenmap(
  bbox = c(left = -0.1218, right = -0.0648, bottom = 51.5033, top = 51.5269),
  maptype = "toner-lite",
  zoom = 15
)

CoL.sf <- st_zm(st_read("CoL.kml"))

ggmap(london.map) +
  geom_sf(data = CoL.sf, 
          inherit.aes = FALSE, 
          fill = alpha("red", 0.05), 
          color="red",
          linetype = "dashed",
          size = 1) +
  geom_point(data = location.df,
             aes(x = Longitude, y = Latitude),
             size = 1,
             color = alpha("blue", 0.5)) +
  theme_map()

CoL.sf <- st_transform(CoL.sf, 27700)
CoL.sp <- as(CoL.sf, "Spatial")
CoL.owin <- as(CoL.sp, "owin")
CoL.owin <- rescale(CoL.owin, 1000)
location.ppp$window <- CoL.owin

L <- Lest(location.ppp, correction="Ripley")
plot(L, .-r~r, main=" ")

estlam = 148/area(CoL.owin)
llll <- numeric(1000)
for (i in 1:1000) {
  aaa <- rpoispp(estlam, win=CoL.owin)
  bbb <- Lest(aaa, correction="Ripley")
  llll[i] <- max(abs(bbb$iso - bbb$r))
  if (i %% 10 == 0) {
    print(i)
  }
}

t <- max(abs(L$iso[100:500]-L$r[100:500]))
c <- quantile(llll, 0.95)

# --------- END --------
