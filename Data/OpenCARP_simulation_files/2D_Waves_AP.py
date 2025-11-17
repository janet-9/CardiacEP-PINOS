#Script to run simple planar and point source waves on the 2D mesh using the AP model
import os
EXAMPLE_DIR = os.path.dirname(__file__)
CALLER_DIR = os.getcwd()
GUIinclude = False

from datetime import date
from carputils import tools
from carputils import ep
from carputils.carpio import txt
import sys


def parser():
    # Generate the standard command line parser
    parser = tools.standard_parser()
    group  = parser.add_argument_group('experiment specific options')

    # Add experiment arguments
    group.add_argument('--mesh',
                       type = str, default = '2D_10cm_250um',
                       help = 'Mesh to run the experiment on (default is %(default)s)')
    group.add_argument('--init',
                       type = str, default = 'datasets_start_states/INIT_500_BCL_AlievPanfilovDynamic_Model.sv',
                       help = 'Single cell initial state (default is %(default)s)')
    group.add_argument('--tend',
                       type = float, default = 1000,
                       help = 'Duration of simulation (default is %(default)s) ms')
    group.add_argument('--pls',
                       type = float, default = 1,
                       help = 'Number of pulses (default is %(default)s)')
    group.add_argument('--bcl',
                        type = float, default = 500,
                        help = 'BCL pulses (default is %(default)s ms, 80bpm )')
    group.add_argument('--prepacing_bcl',
                        type = float, default = 500,
                        help = 'Pre-pacing BCL pulses (default is %(default)s ms, 80bpm )')
    group.add_argument('--prepacing_stimstr',
                        type = float, default = 50,
                        help = 'Strength for the pre-pacing stimulation (default is %(default)s muA)')
    group.add_argument('--prepacing_stimdur',
                        type = float, default = 2,
                        help = 'Duration for the stimulation (default is %(default)s  ms)')
    group.add_argument('--strength',
                        type = float, default = 50,
                        help = 'Strength for the stimulation (default is %(default)s muA)')
    group.add_argument('--duration',
                        type = float, default = 2,
                        help = 'Duration for the stimulation (default is %(default)s  ms)')
    group.add_argument('--model',
                        type = str,
                        default = "AlievPanfilovDynamic", 
                        help='ionic model (default is %(default)s)')
    group.add_argument('--output_res',
                       type = float, default = 5,
                       help = 'Resolution of output for .igb files (default is %(default)s ms)')
    group.add_argument('--check',
                       type = float, default = 1000,
                       help = 'Time interval for saving the state of the simulation (default is %(default)s ms, 1 second )')
    group.add_argument('--aniso',
                       action = 'store_true',
                       help = 'Flag to us anisotropic conductivity values')
    group.add_argument('--conmul',
                       type = float,
                       default = 1.0,
                       help = 'Multiplier for myocardium conductivities, default is %(default)s')

    # Protocol Arguments 
    group.add_argument('--protocol',
                       choices = ['planar', 'centrifugal', 'spiral_init', 'spiral', 'breakup'],
                       default = 'planar',
                       help = 'Option for simple wave excitement, choose from %(choices)s, default is %(default)s, propagating left to right')
    group.add_argument('--start_state', 
                       default = 'Spiral_250um_1000.0ms.roe',
                       help = 'Starting state for continued simulation, default is %(default)s')
    group.add_argument('--K', 
                       default = 1.0, type = float, 
                       help = 'Scaling Parameter value for K, default is %(default)s')
    group.add_argument('--a', 
                       default = 1.0, type = float,
                       help = 'Scaling Parameter value for a, default is %(default)s')
    group.add_argument('--b', 
                       default = 1.0, type = float,
                       help = 'Scaling Parameter value for b, default is %(default)s')
    group.add_argument('--epsilon', 
                       default = 1.0, type = float,
                       help = 'Scaling Parameter value for epsilon, default is %(default)s')
    group.add_argument('--mu1', 
                       default = 1.0, type = float,
                       help = 'Scaling Parameter value for mu1, default is %(default)s')
    group.add_argument('--mu2', 
                       default = 1.0, type = float,
                       help = 'Scaling Parameter value for mu2, default is %(default)s')
    group.add_argument('--second_stim_start', 
                       default = 325,
                       help= 'Start time for second stimulus for cross stimulus spiral propagation, default = %(default)s')
    group.add_argument('--second_stim_pls',
                       default = 1,
                       help = 'Number of second stimulus pulses in cross stimulus spiral propagation, default is %(default)s')
    group.add_argument('--second_stim_bcl',
                       default = 500,
                       help = 'BCL of second stimulus pulses in cross stimulus spiral propagation, default is %(default)s')
   

    return parser

def jobID(args):
    today = date.today()


    #Adjust the imp parameters according to the scaling arguments 
    K_val = 8*args.K
    a_val = 0.15*args.a
    b_val = 0.15*args.b
    epsilon_val = 0.002*args.epsilon
    mu1_val = 0.2*args.mu1
    mu2_val = 0.3*args.mu2

    if args.aniso:
        aniso_flag = 1
    else: 
        aniso_flag = 0


    if args.protocol == 'spiral_init':
        ID = '{}_2D_spiral_INIT_{}_{}_{}_bcl_{}_pls_{}_uA_{}_ms_{}_start_{}_K_{}_a_{}_b_{}_epsilon_{}_mu1_{}_mu2_{}_tend_{}'.format(today.isoformat(), args.mesh, args.model, args.bcl,
                                        args.pls, args.strength, args.duration, args.second_stim_start, K_val, a_val, b_val, epsilon_val, mu1_val,
                                        mu2_val, args.tend, aniso_flag)
        return ID
    elif args.protocol == 'spiral':
        ID = '{}_2D_spiral_{}_{}_{}_bcl_{}_pls_{}_uA_{}_ms_{}_start_{}_K_{}_a_{}_b_{}_epsilon_{}_mu1_{}_mu2_{}_conmul_{}_tend_{}'.format(today.isoformat(), args.mesh, args.model, args.bcl,
                                        args.pls, args.strength, args.duration, args.second_stim_start, K_val, a_val, b_val, epsilon_val, mu1_val,
                                        mu2_val, args.conmul, args.tend, aniso_flag)
        return ID
    elif args.protocol == 'centrifugal':
          ID = '{}_2D_centrifugal_{}_{}_{}_bcl_{}_pls_{}_uA_{}_ms_{}_K_{}_a_{}_b_{}_epsilon_{}_mu1_{}_mu2_{}_conmul_{}_tend_{}'.format(today.isoformat(), args.mesh, args.model, args.bcl,
                                        args.pls, args.strength, args.duration, K_val, a_val, b_val, epsilon_val, mu1_val,
                                        mu2_val, args.conmul, args.tend, aniso_flag)
          return ID
    elif args.protocol == 'breakup':
         ID = '{}_2D_spiral_breakup_{}_{}_{}_K_{}_a_{}_b_{}_epsilon_{}_mu1_{}_mu2_{}_conmul_{}_tend_{}'.format(today.isoformat(), args.mesh, args.model, K_val, a_val, b_val, epsilon_val, mu1_val,
                                        mu2_val, args.conmul, args.tend, aniso_flag)
         return ID
    else:
          ID = '{}_2D_planar_{}_{}_{}_bcl_{}_pls_{}_uA_{}_ms_{}_K_{}_a_{}_b_{}_epsilon_{}_mu1_{}_mu2_{}_conmul_{}_tend_{}'.format(today.isoformat(), args.mesh, args.model, args.bcl,
                                        args.pls, args.strength, args.duration, K_val, a_val, b_val, epsilon_val, mu1_val,
                                        mu2_val, args.conmul, args.tend, aniso_flag)
          return ID      

@tools.carpexample(parser, jobID)



def run(args, job):
    
        # Generate general command line
        cmd = tools.carp_cmd()

        # Set output directory
        cmd += ['-simID', job.ID]

        import os, sys

        # Ensure the job folder exists
        os.makedirs(job.ID, exist_ok=True)

        log_path = os.path.join(job.ID, "simulation_output.dat")
        logfile = open(log_path, "w")

        class Tee:
                def __init__(self, *files):
                        self.files = files
                def write(self, data):
                        for f in self.files:
                                f.write(data)
                def flush(self):
                        for f in self.files:
                                f.flush()

        sys.stdout = Tee(sys.stdout, logfile)
        sys.stderr = Tee(sys.stderr, logfile)


        # Add some example-specific command line options
        cmd += ['-meshname', args.mesh,
                        '-tend', args.tend ,
                        '-gridout_i', 3,
                        '-gridout_e', 3
                ]
        #Define the physics regions
        cmd += ['-num_phys_regions', 2,
                        '-phys_region[0].name', 'intracellular',
                        '-phys_region[0].ptype', 0,
                        '-phys_region[1].name', 'extracellular',
                        '-phys_region[1].ptype', 1
                        ]
        
        cmd += ['-num_imp_regions',          1,
                '-imp_region[0].im',         args.model
        ]

        #Define outputs and postprocesses, save the final state
        cmd += ['-spacedt', args.output_res,
                        '-timedt', 1.0,
                        '-compute_APD', 1,
                        #'-num_tsav', 1,
                        #'-tsav', args.tend
                        ]
        
        # Save the checkpoints of the simulation 
        cmd += ['-chkpt_start', 0,
                '-chkpt_intv',  args.check,
                '-chkpt_stop', args.tend
                ]
        
        #Define the outputs via trace files
        cmd += ['-num_gvecs', 1,
                '-gvec[0].imp', args.model,
                '-gvec[0].ID[0]', 'V',
                '-gvec[0].name', 'w', 
                        ]

        # Set monodomain conductivities for the tissue and the bath - adjust conditions for spiral breakup if needed
        # Toggle for isotropic vs anisotropic conduction for the slab (2D)

        if args.aniso:
                cmd += [ '-num_gregions',	1,
                                
                        '-gregion[0].name', 		"myocardium",
                        '-gregion[0].num_IDs',           1,
                        '-gregion[0].ID[0]', 		"100",		

                        # mondomain conductivites will be calculated as half of the harmonic mean of intracellular
                        # and extracellular conductivities

                        '-gregion[0].g_il',       0.174,
                        '-gregion[0].g_el',       0.625,
                        '-gregion[0].g_it',       0.019,
                        '-gregion[0].g_et',	  0.236,
                        '-gregion[0].g_in',       0.019,
                        '-gregion[0].g_en',	  0.236,
                        '-gregion[0].g_mult',	  2.5 * args.conmul,

                                        ]
        else:
                cmd += [ '-num_gregions',			1,
                        
                        '-gregion[0].name', 		"myocardium",
                        '-gregion[0].num_IDs',           1,
                        '-gregion[0].ID[0]', 		"100",		

                        # mondomain conductivites will be calculated as half of the harmonic mean of intracellular
                        # and extracellular conductivities

                        '-gregion[0].g_il',       0.174,
                        '-gregion[0].g_el',       0.625,
                        '-gregion[0].g_it',       0.174,
                        '-gregion[0].g_et',	  0.625,
                        '-gregion[0].g_in',       0.174,
                        '-gregion[0].g_en',	  0.625,
                        '-gregion[0].g_mult',	  2.5 * args.conmul,

                                ]

        if args.protocol == 'spiral' or args.protocol == 'breakup':
                
                #Define the starting state
                cmd += [
                        '-start_statef', args.start_state
                        ]
                
                #Adjust the imp parameters for inducing spiral breakup, scaling the default values 
                K_val = args.K
                a_val = args.a
                b_val = args.b
                epsilon_val = args.epsilon
                mu1_val = args.mu1
                mu2_val = args.mu2

                cmd += [
                        '-imp_region[0].im_param', f'K*{K_val},a*{a_val},b*{b_val},epsilon*{epsilon_val},mu1*{mu1_val},mu2*{mu2_val}',
                        ]

                # Run example
                job.carp(cmd)


        else:
                
                #Define the initial stages for the model 
                cmd += [
                        '-imp_region[0].im_sv_init', args.init
                        ]
                
                #Define the stimulus depending on the protocol

                # Prepace the model
                cmd += ['-prepacing_beats', 4,
                        '-prepacing_bcl', args.prepacing_bcl,
                        #'-prepacing_stimdur', args.prepacing_stimdur,
                        #'-prepacing_stimstr', args.prepacing_stimstr
                        ]


                if args.protocol == 'spiral_init':

                        #Adjust the imp parameters for inducing spiral breakup, scaling the default values 
                        K_val = args.K
                        a_val = args.a
                        b_val = args.b
                        epsilon_val = args.epsilon
                        mu1_val = args.mu1
                        mu2_val = args.mu2

                        cmd += [
                                '-imp_region[0].im_param', f'K*{K_val},a*{a_val},b*{b_val},epsilon*{epsilon_val},mu1*{mu1_val},mu2*{mu2_val}',
                                ]
                #Set up electrode across the left wall and dump the electrode to a vtx file.             
                        cmd += ['-num_stim',  2,
                                
                                # Define the electrode geometry based on the mesh (all along the left edge, thin in the x direction)

                                # Define the stimulus pulse 
                                '-stimulus[0].name', 'left_wall'
                                '-stimulus[0].stimtype',   0,
                                '-stimulus[0].strength',   args.strength,
                                '-stimulus[0].duration',   args.duration,
                                '-stimulus[0].npls',       args.pls,
                                '-stimulus[0].bcl',        args.bcl,
                                #'-stimulus[0].geometry', 100,

                                #Define electrode based on mesh geomtery (left wall)
                                #'-stimulus[1].ctr_def', 1,
                                '-stimulus[0].x0', -50000,
                                '-stimulus[0].y0', -50000,
                                '-stimulus[0].z0', 0,

                                '-stimulus[0].xd', 5000,
                                '-stimulus[0].yd', 100000,
                                '-stimulus[0].zd', 0,
                                

                                # Dump the electrode file
                                '-stimulus[0].dump_vtx_file', 1,


                                # Define the electrode geometry based on the mesh (bottom quadrant of the mesh)


                                # Define the stimulus pulse 
                                '-stimulus[1].name', 'second_electrode'
                                '-stimulus[1].stimtype',   0,
                                '-stimulus[1].strength',   args.strength,
                                '-stimulus[1].duration',   args.duration,
                                '-stimulus[1].npls',       args.second_stim_pls,
                                '-stimulus[1].bcl',        args.second_stim_bcl,
                                '-stimulus[1].start',       args.second_stim_start,
                                #'-stimulus[1].geometry',     200,    

                                # Define the electrode geometry based on the mesh (bottom half of the mesh)
                                #'-stimulus[1].ctr_def', 1,
                                '-stimulus[1].x0', -45000,
                                '-stimulus[1].y0', -45000,
                                '-stimulus[1].z0', 0,

                                '-stimulus[1].xd', 5000,
                                '-stimulus[1].yd', 80000,
                                '-stimulus[1].zd', 0,
                                

                                # Dump the electrode file
                                '-stimulus[1].dump_vtx_file', 1

                        ]
                        
                elif args.protocol == 'centrifugal':
                # Pulse from a small patch in the bottom corner 
                        cmd += [    '-num_stim',  1,

                                # Define the stimulus pulse 
                                '-stimulus[0].stimtype',   0,
                                '-stimulus[0].strength', args.strength,
                                '-stimulus[0].duration',   args.duration,
                                '-stimulus[0].npls',       args.pls,
                                
                                # Define the electrode geometry based on the mesh 
                                #'-stimulus[0].ctr_def', 1,
                                '-stimulus[0].x0', -50000,
                                '-stimulus[0].y0', -50000,
                                '-stimulus[0].z0', 0,

                                '-stimulus[0].xd', 25000,
                                '-stimulus[0].yd', 25000,
                                '-stimulus[0].zd', 0,
                                

                                # Dump the electrode file
                                '-stimulus[0].dump_vtx_file', 1
                                ]
                else:
                #Default Planar Wave: Set up electrode across the left wall and dump the electrode to a vtx file.             
                        cmd += ['-num_stim',  1,
                                

                                # Define the stimulus pulse 
                                '-stimulus[0].stimtype',   0,
                                '-stimulus[0].strength', args.strength,
                                '-stimulus[0].duration',   args.duration,
                                '-stimulus[0].npls',       args.pls,
                                #'-stimulus[0].geometry', 100,

                                # Dump the electrode file
                                '-stimulus[0].dump_vtx_file', 1,

                                #Define electrode based on mesh geomtery (left wall)
                                #'-stimulus[1].ctr_def', 1,
                                '-stimulus[0].x0', -50000,
                                '-stimulus[0].y0', -50000,
                                '-stimulus[0].z0', 0,

                                '-stimulus[0].xd', 1000,
                                '-stimulus[0].yd', 100000,
                                '-stimulus[0].zd', 0,

                        ]       

                #Calculate the APDs for 90% repolarization:
                cmd += ['-num_LATs', 2, 
                        
                        '-lats[0].ID ', 'LATS',
                        '-lats[0].all', 0,
                        '-lats[0].measurand', 0,
                        '-lats[0].threshold',  '-20',
                        '-lats[0].mode', 0,

                        '-lats[1].ID',         'REPS',
                        '-lats[1].all',         0,
                        '-lats[1].measurand',  0,
                        '-lats[1].threshold', '-70',
                        '-lats[1].mode', 1,
                        ]

                cmd += ['-compute_APD', 1,
                        '-actthresh', '-10', 
                        '-recovery_thresh', '-70'
                        ]
        
                # Run example
                job.carp(cmd)

                # Calculate the APDs:
                LATs = txt.read(os.path.join(job.ID, 'init_acts_LATS-thresh.dat'))
                REPs = txt.read(os.path.join(job.ID, 'init_acts_REPS-thresh.dat'))
                APDs = REPs - LATs
                txt.write(os.path.join(job.ID, 'APDs.dat'), APDs)

if __name__ == '__main__':
    run()