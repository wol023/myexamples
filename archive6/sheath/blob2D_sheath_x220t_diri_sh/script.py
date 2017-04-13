# Define expressions
DefineScalarExpression("delta n", "(component_0-conn_cmfe(</global/u2/w/wol023/linkToCoriScratch1.ln/eslexamples/myexamples/tanhbol_64x64x32x16_dt005_isolated_newvlasov_20_te5_by06/plt_density_plots/plt.1.hydrogen.density0000.2d.hdf5:component_0>, Mesh) )")
DefineVectorExpression("vector3D", "{component_0,component_1,component_2}")
DefineVectorExpression("Bvec3D", "{component_0,component_2,-component_1}")
DefineVectorExpression("vector2D", "{component_0,component_1}")
DefineVectorExpression("Bvec2D", "{component_0,component_2}")
DefineVectorExpression("vector_xy0", "{component_0,component_1,component_2*0.0}")
# Write color tables
SetCloneWindowOnFirstRef(0)
###############################################################################
width, height = 669, 609
win = GetGlobalAttributes().windows[GetGlobalAttributes().activeWindow]
ResizeWindow(win, width, height)
SetActiveWindow(win) # Synchronize
size = GetWindowInformation().windowSize
if width < size[0] or height < size[1]:
    ResizeWindow(win, width + (size[0] - width), height + (size[1] - height))
DeleteAllPlots()
for name in GetAnnotationObjectNames():
    DeleteAnnotationObject(name)

# Create plots
# Create plot 1
OpenDatabase("localhost:/home/wonjae/ws/myexamples/sheath/blob2D_sheath_x220t_diri_sh/plt_potential_plots/plt.potential*.hdf5 database")
AddPlot("Pseudocolor", "component_0", 0, 0)
atts = PseudocolorAttributes()
#SetPlotOptions(atts)
silr = SILRestriction()
silr.TurnOnAll()
SetPlotSILRestriction(silr, 0)

SetActivePlots(0)

DrawPlots()

# Set the view
view = View2DAttributes()
view.windowCoords = (0, 6.28319, 0, 6.28319)
view.viewportCoords = (0.2, 0.95, 0.15, 0.95)
view.fullFrameActivationMode = view.Auto  # On, Off, Auto
view.fullFrameAutoThreshold = 100
view.xScale = view.LINEAR  # LINEAR, LOG
view.yScale = view.LINEAR  # LINEAR, LOG
view.windowValid = 1
SetView2D(view)

# Set the annotation attributes
annot = AnnotationAttributes()
#SetAnnotationAttributes(annot)

# Set annotation object properties
win0_legend000 = GetAnnotationObject(GetPlotList().GetPlots(0).plotName)
win0_legend000.active = 1
win0_legend000.managePosition = 1
win0_legend000.position = (0.05, 0.9)
win0_legend000.xScale = 1
win0_legend000.yScale = 1
win0_legend000.textColor = (0, 0, 0, 255)
win0_legend000.useForegroundForTextColor = 1
win0_legend000.drawBoundingBox = 0
win0_legend000.boundingBoxColor = (0, 0, 0, 50)
win0_legend000.numberFormat = "%# -9.4g"
win0_legend000.fontFamily = win0_legend000.Arial  # Arial, Courier, Times
win0_legend000.fontBold = 0
win0_legend000.fontItalic = 0
win0_legend000.fontShadow = 0
win0_legend000.fontHeight = 0.015
win0_legend000.drawLabels = win0_legend000.Values # None, Values, Labels, Both
win0_legend000.drawTitle = 1
win0_legend000.drawMinMax = 1
win0_legend000.orientation = win0_legend000.VerticalRight  # VerticalRight, VerticalLeft, HorizontalTop, HorizontalBottom
win0_legend000.controlTicks = 1
win0_legend000.numTicks = 5
win0_legend000.minMaxInclusive = 1
win0_legend000.suppliedValues = (-1.72292e-14, -1.34238e-14, -9.61837e-15, -5.81295e-15, -2.00752e-15)
win0_legend000.suppliedLabels = ()

SetActiveWindow(GetGlobalAttributes().windows[0])
