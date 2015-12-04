# Visit 2.10.0 log file
ScriptVersion = "2.10.0"
if ScriptVersion != Version():
    print "This script is for VisIt %s. It may not work with version %s" % (ScriptVersion, Version())
ShowAllWindows()
DefineVectorExpression("e-vector", "{component_0,component_1}")
OpenDatabase("localhost:/Users/lee1041/ws/myexamples/drift_blob/plt_efield_plots/plt.efield*.hdf5 database", 0)
metadata = GetMetaData("localhost:/Users/lee1041/ws/myexamples/drift_blob/plt_efield_plots/plt.efield*.hdf5 database", -1)
AddPlot("Mesh", "Mesh", 1, 1)
HideActivePlots()
AddPlot("Pseudocolor", "component_0", 1, 1)
DrawPlots()
DeleteActivePlots()
DeleteActivePlots()
AddPlot("Vector", "e-vector", 1, 1)
DrawPlots()
TimeSliderNextState()
TimeSliderNextState()
TimeSliderNextState()
TimeSliderNextState()
TimeSliderNextState()
TimeSliderNextState()
TimeSliderNextState()
TimeSliderNextState()
TimeSliderNextState()
TimeSliderNextState()
TimeSliderNextState()
TimeSliderNextState()
TimeSliderNextState()
TimeSliderNextState()
TimeSliderNextState()
TimeSliderNextState()
TimeSliderNextState()
# Begin spontaneous state
View2DAtts = View2DAttributes()
View2DAtts.windowCoords = (-0.189373, 6.09381, 0.468448, 6.75163)
View2DAtts.viewportCoords = (0.2, 0.95, 0.15, 0.95)
View2DAtts.fullFrameActivationMode = View2DAtts.Auto  # On, Off, Auto
View2DAtts.fullFrameAutoThreshold = 100
View2DAtts.xScale = View2DAtts.LINEAR  # LINEAR, LOG
View2DAtts.yScale = View2DAtts.LINEAR  # LINEAR, LOG
View2DAtts.windowValid = 1
SetView2D(View2DAtts)
# End spontaneous state

WriteConfigFile()
