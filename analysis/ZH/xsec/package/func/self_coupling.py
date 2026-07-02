from abc import ABCMeta, abstractmethod



###############################
### HELPER CLASS DEFINITION ###
###############################

### Class that takes care of building a physics model by combining individual channels and processes together
### Things that it can do:
###   - define the parameters of interest (in the default implementation , "r")
###   - define other constant model parameters (e.g., "MH")
###   - yields a scaling factor for each pair of bin and process (by default, constant for background and linear in "r" for signal)
###   - possibly modifies the systematical uncertainties (does nothing by default)

class PhysicsModelBase(metaclass=ABCMeta):
    def __init__(self):
        pass

    def setModelBuilder(self, modelBuilder):
        "Connect to the ModelBuilder to get workspace, datacard and options. Should not be overloaded."
        self.modelBuilder = modelBuilder
        self.DC = modelBuilder.DC
        self.options = modelBuilder.options

    def setPhysicsOptions(self, physOptions):
        "Receive a list of strings with the physics options from command line"

    @abstractmethod
    def doParametersOfInterest(self):
        """Create POI and other parameters, and define the POI set."""

    def preProcessNuisances(self, nuisances):
        "receive the usual list of (name,nofloat,pdf,args,errline) to be edited"
        pass  # do nothing by default

    def getYieldScale(self, bin, process):
        "Return the name of a RooAbsReal to scale this yield by or the two special values 1 and 0 (don't scale, and set to zero)"
        return "r" if self.DC.isSignal[process] else 1

    def getChannelMask(self, bin):
        "Return the name of a RooAbsReal to mask the given bin (args != 0 => masked)"
        name = "mask_%s" % bin
        # Check that the mask expression doesn't exist already, it might do
        # if it was already defined in the datacard
        if not self.modelBuilder.out.arg(name):
            self.modelBuilder.doVar("%s[0]" % name)
        return name

    def done(self):
        "Called after creating the model, except for the ModelConfigs"
        pass



############################
### CLASS BASE DEFINTION ###
############################

class PhysicsModel(PhysicsModelBase):
    """Example class with signal strength as only POI"""

    def doParametersOfInterest(self):
        """Create POI and other parameters, and define the POI set."""
        self.modelBuilder.doVar("r[1,0,20]")
        self.modelBuilder.doSet("POI", "r")
        # --- Higgs Mass as other parameter ----
        if self.options.mass != 0:
            if self.modelBuilder.out.var("MH"):
                var = self.modelBuilder.out.var("MH")
                var.removeMin()
                var.removeMax()
                var.setVal(self.options.mass)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass)
