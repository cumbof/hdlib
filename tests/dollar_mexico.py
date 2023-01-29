from hdlib.space import Space, Vector
from hdlib.arithmetic import bind, bundle, permute

# Kanerva, P., 2010, November. 
# What we mean when we say" What's the dollar of Mexico?": Prototypes and mapping in concept space. 
# In 2010 AAAI fall symposium series.

# Initialize vectors space
space = Space()

# Define features and country information
names = [
  "NAM", "CAP", "MON", # Features
  "USA", "WDC", "DOL", # United States of America
  "MEX", "MXC", "PES"  # Mexico
]

# Build a random bipolar vector for each feature and country information
# Add vectors to the space
space.bulkInsert(names)

# Encode USA information in a single vector
# USTATES = [(NAM * USA) + (CAP * WDC) + (MON * DOL)]
USTATES_NAM = bind(space.get("NAM"), space.get("USA")) # Bind NAM with USA
USTATES_CAP = bind(space.get("CAP"), space.get("WDC")) # Bind CAP with WDC
USTATES_MON = bind(space.get("MON"), space.get("DOL")) # Bind MON with DOL
USTATES = bundle(bundle(USTATES_NAM, USTATES_CAP), USTATES_MON) # Bundle USTATES_NAM, USTATES_CAP, and USTATES_MON

# Repeat the last step to encode MEX information in a single vector
# MEXICO = [(NAM * MEX) + (CAP * MXC) + (MON * PES)]
MEXICO_NAM = bind(space.get("NAM"), space.get("MEX")) # Bind NAM with MEX
MEXICO_CAP = bind(space.get("CAP"), space.get("MXC")) # Bind CAP with MXC
MEXICO_MON = bind(space.get("MON"), space.get("PES")) # Bind MON with PES
MEXICO = bundle(bundle(MEXICO_NAM, MEXICO_CAP), MEXICO_MON) # Bundle MEXICO_NAM, MEXICO_CAP, and MEXICO_MON

# F_UM = USTATES * MEXICO
#      = [(USA * MEX) + (WDC * MXC) + (DOL * PES) + noise]
F_UM = bind(USTATES, MEXICO)

# DOL * F_UM = DOL * [(USA * MEX) + (WDC * MXC) + (DOL * PES) + noise]
#            = [(DOL * USA * MEX) + (DOL * WDC * MXC) + (DOL * DOL * PES) + (DOL * noise)]
#            = [noise1 + noise2 + PES + noise3]
#            = [PES + noise4]
#            ≈ PES
GUESS_PES = bind(space.get("DOL"), F_UM)

print(space.find(GUESS_PES))